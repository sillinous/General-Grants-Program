#!/usr/bin/env python3
"""
MCP Tools Indexer

Discovers MCP servers under a GitHub org (default: modelcontextprotocol),
recursively crawls README GitHub links (depth-limited), optionally performs
global code searches, detects likely servers via topics and manifests, and
extracts registered tools via lightweight parsing. Produces a consolidated
index (JSON + Markdown).

Requires network access. Set GITHUB_TOKEN for higher rate limits.

Usage:
  python scripts/mcp_tools_index.py --org modelcontextprotocol \
    --crawl-links --crawl-depth 2 --search-global \
    --max-repos 300 \
    --out-json mcp_tools_index.json --out-md MCP_TOOLS_INDEX.md

Notes:
- Detection uses GitHub code search heuristics and best-effort regex parsing.
- Crawling follows README links to other GitHub repos within a depth limit.
- Global code search is rate-limited; provide a token to improve reliability.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple
import urllib.error

import urllib.parse
import urllib.request
import ssl


GITHUB_API = "https://api.github.com"


def gh_request(path: str, params: Optional[Dict[str, str]] = None) -> dict:
    url = f"{GITHUB_API}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url)
    token = os.getenv("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "mcp-tools-indexer/0.1")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def gh_request_paginated(path: str, params: Optional[Dict[str, str]] = None) -> List[dict]:
    items: List[dict] = []
    page = 1
    while True:
        merged = dict(params or {})
        merged.setdefault("per_page", 100)
        merged["page"] = page
        data = gh_request(path, merged)
        if not isinstance(data, list):
            break
        items.extend(data)
        if len(data) < int(merged["per_page"]):
            break
        page += 1
    return items


def list_org_repos(org: str) -> List[dict]:
    try:
        repos = gh_request_paginated(f"/orgs/{org}/repos", params={"type": "public", "sort": "full_name"})
    except urllib.error.HTTPError as e:
        if e.code == 403:
            # Fallback to search API to list repos by org when core rate limit is exhausted
            data = gh_request("/search/repositories", params={"q": f"org:{org} fork:false", "per_page": 100})
            repos = data.get("items", [])
        else:
            raise
    # Filter out archived/forks/disabled
    out = []
    for r in repos:
        if r.get("archived") or r.get("disabled") or r.get("fork"):
            continue
        out.append(r)
    return out


def search_code(repo_full: str, query: str) -> List[dict]:
    q = f"{query} repo:{repo_full}"
    data = gh_request("/search/code", params={"q": q, "per_page": 100})
    items = data.get("items", [])
    return items


def fetch_file_contents(repo_full: str, ref: str, path: str) -> Optional[str]:
    # Use raw content endpoint for simplicity
    url = f"https://raw.githubusercontent.com/{repo_full}/{ref}/{path}"
    req = urllib.request.Request(url)
    token = os.getenv("GITHUB_TOKEN")
    if token:
        # Raw endpoints don't accept auth headers for private files; but this is a public org.
        pass
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def get_repo(repo_full: str) -> Optional[dict]:
    try:
        return gh_request(f"/repos/{repo_full}")
    except Exception:
        return None


def get_repo_topics(repo_full: str) -> List[str]:
    try:
        data = gh_request(f"/repos/{repo_full}/topics")
        names = data.get("names", [])
        return [str(n) for n in names]
    except Exception:
        return []


def try_fetch_readme(repo_full: str, ref: str) -> Optional[str]:
    candidates = [
        "README.md",
        "Readme.md",
        "readme.md",
        "README.MD",
        "README",
        "README.rst",
    ]
    for name in candidates:
        src = fetch_file_contents(repo_full, ref, name)
        if src:
            return src
    return None


# --- Heuristic parsers -------------------------------------------------------

JS_TOOL_OBJ_RE = re.compile(
    r"server\.(?:tool|addTool)\s*\(\s*\{(?P<obj>[^}]+)\}\s*(?:,|\))",
    re.DOTALL,
)
JS_PROP_RE = re.compile(r'(name|description)\s*:\s*([`\'\"])(.*?)\2', re.DOTALL)


def parse_js_ts_tools(src: str) -> List[Dict[str, str]]:
    tools: List[Dict[str, str]] = []
    for m in JS_TOOL_OBJ_RE.finditer(src):
        obj = m.group("obj")
        found = {k: v for (k, v) in JS_PROP_RE.findall(obj)}
        if "name" in found:
            tools.append({
                "name": found.get("name", "").strip(),
                "description": found.get("description", "").strip(),
                "language": "js/ts",
            })
    return tools


PY_DECORATOR_RE = re.compile(r"@\s*(?:server\.)?tool\s*(?:\((?P<args>[^)]*)\))?", re.DOTALL)
PY_DEF_RE = re.compile(r"def\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")


def extract_first_docstring(src: str, def_pos: int) -> str:
    # naive: look for triple-quote docstring after def
    post = src[def_pos:]
    m = re.search(r"^[ \t]*\"\"\"(.*?)\"\"\"|^[ \t]*\'\'\'(.*?)\'\'\'", post, re.DOTALL | re.MULTILINE)
    if not m:
        return ""
    doc = m.group(1) or m.group(2) or ""
    # compress whitespace
    return re.sub(r"\s+", " ", doc).strip()


def parse_python_tools(src: str) -> List[Dict[str, str]]:
    tools: List[Dict[str, str]] = []
    for dm in PY_DECORATOR_RE.finditer(src):
        start = dm.end()
        # find next def
        dm_def = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", src[start:])
        if not dm_def:
            continue
        func_name = dm_def.group(1)
        func_pos = start + dm_def.start()
        # Try to capture explicit name= in decorator args
        args = dm.group("args") or ""
        m_name = re.search(r"name\s*=\s*([\'\"])(.*?)\1", args)
        tool_name = m_name.group(2).strip() if m_name else func_name
        desc = extract_first_docstring(src, func_pos)
        tools.append({
            "name": tool_name,
            "description": desc,
            "language": "python",
        })
    return tools


def dedupe_tools(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out = []
    for it in items:
        key = (it.get("name", ""), it.get("language", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


ENDPOINT_URL_RE = re.compile(r"(wss?://[^\s)\"'<>]+|https?://[^\s)\"'<>]+)", re.IGNORECASE)


def extract_endpoints_from_text(text: str) -> List[str]:
    urls = [m.group(1) for m in ENDPOINT_URL_RE.finditer(text or "")]
    # Prefer likely MCP endpoints
    ranked = []
    for u in urls:
        score = 0
        lu = u.lower()
        if any(k in lu for k in ("/mcp", "/sse", "mcp", "/server")):
            score += 2
        if lu.startswith("wss://") or lu.startswith("ws://"):
            score += 1
        ranked.append((score, u))
    ranked.sort(reverse=True)
    # Dedupe preserving order
    out: List[str] = []
    seen = set()
    for _, u in ranked:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def check_http_url(url: str, timeout: float = 5.0) -> Tuple[bool, Optional[int]]:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return True, getattr(resp, 'status', None)
    except Exception:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return True, getattr(resp, 'status', None)
        except Exception:
            return False, None


def detect_and_extract_tools(repo: dict, verify_endpoints: bool = False) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[str], List[Dict[str, Optional[str]]]]:
    repo_full = repo["full_name"]
    default_branch = repo.get("default_branch", "main")
    server_hits: List[dict] = []
    tool_defs: List[Dict[str, str]] = []
    hints: List[str] = []
    endpoints: List[Dict[str, Optional[str]]] = []

    # Queries to find likely MCP servers / tool registration points
    queries = [
        # JS/TS MCP SDK usage
        '"@modelcontextprotocol/sdk"',
        'server.tool(',
        'addTool(',
        # Python MCP usage
        '"modelcontextprotocol"+tool',
        '@tool',
        'server.tool(',
    ]

    matched_files = {}
    for q in queries:
        try:
            results = search_code(repo_full, q)
        except Exception:
            # Back off briefly on API errors
            time.sleep(1)
            continue
        for item in results:
            path = item.get("path")
            if not path:
                continue
            matched_files[path] = item

    for path, item in matched_files.items():
        # Only consider source-like files
        if not any(path.endswith(ext) for ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".py")):
            continue
        src = fetch_file_contents(repo_full, default_branch, path)
        if not src:
            continue
        local_tools: List[Dict[str, str]] = []
        if path.endswith((".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")):
            local_tools = parse_js_ts_tools(src)
        elif path.endswith(".py"):
            local_tools = parse_python_tools(src)
        if local_tools:
            server_hits.append({
                "path": path,
                "tools_count": len(local_tools),
            })
            tool_defs.extend(local_tools)

    # Manifest/topic-based hints
    pkg_json = fetch_file_contents(repo_full, default_branch, "package.json")
    if pkg_json:
        try:
            pkg = json.loads(pkg_json)
        except Exception:
            pkg = {}
        deps = {}
        for k in ("dependencies", "devDependencies", "peerDependencies"):
            d = pkg.get(k)
            if isinstance(d, dict):
                deps.update(d)
        if any(name.startswith("@modelcontextprotocol/") for name in deps.keys()) or (
            "@modelcontextprotocol/sdk" in deps
        ):
            hints.append("package.json: mcp sdk dependency")

    req_txt = fetch_file_contents(repo_full, default_branch, "requirements.txt")
    if req_txt and re.search(r"modelcontextprotocol|\bmcp\b", req_txt, re.IGNORECASE):
        hints.append("requirements.txt: modelcontextprotocol/mcp present")

    pyproject = fetch_file_contents(repo_full, default_branch, "pyproject.toml")
    if pyproject and re.search(r"modelcontextprotocol|\bmcp\b", pyproject, re.IGNORECASE):
        hints.append("pyproject.toml: modelcontextprotocol/mcp present")

    topics = get_repo_topics(repo_full)
    if topics:
        if any(t in {"mcp", "mcp-server", "modelcontextprotocol"} for t in topics):
            hints.append("repo topics indicate MCP")

    # Endpoint discovery: README first
    readme = try_fetch_readme(repo_full, default_branch)
    if readme:
        for url in extract_endpoints_from_text(readme):
            status = None
            reachable = None
            if url.lower().startswith("http://") or url.lower().startswith("https://"):
                ok, code = check_http_url(url)
                reachable = ok
                status = str(code) if code is not None else None
            ep = {
                "url": url,
                "source": "README",
                "reachable": reachable,
                "http_status": status,
            }
            endpoints.append(ep)

    # Endpoint discovery: code search for likely mentions
    endpoint_queries = [
        '"wss://" mcp',
        '"ws://" mcp',
        '"https://" mcp',
        'mcp server url',
        'mcp sse',
    ]
    paths = {}
    for q in endpoint_queries:
        try:
            for item in search_code(repo_full, q):
                p = item.get("path")
                if p:
                    paths[p] = item
        except Exception:
            time.sleep(0.5)
            continue
    for p in paths.keys():
        if not any(p.endswith(ext) for ext in (".md", ".ts", ".js", ".py", ".json", ".txt")):
            continue
        src = fetch_file_contents(repo_full, default_branch, p)
        if not src:
            continue
        for url in extract_endpoints_from_text(src):
            status = None
            reachable = None
            if url.lower().startswith("http://") or url.lower().startswith("https://"):
                ok, code = check_http_url(url)
                reachable = ok
                status = str(code) if code is not None else None
            ep = {
                "url": url,
                "source": p,
                "reachable": reachable,
                "http_status": status,
            }
            endpoints.append(ep)

    # Dedupe endpoints by URL while keeping first source/metadata
    deduped_eps: List[Dict[str, Optional[str]]] = []
    seen_ep = set()
    for ep in endpoints:
        u = ep.get("url")
        if u and u not in seen_ep:
            seen_ep.add(u)
            deduped_eps.append(ep)

    # Optional verification for ws/wss endpoints (fetch tools via JSON-RPC)
    if verify_endpoints:
        try:
            import websocket  # type: ignore
        except Exception:
            websocket = None  # type: ignore
        if websocket is not None:
            for ep in deduped_eps:
                url = (ep.get("url") or "").lower()
                if url.startswith("ws://") or url.startswith("wss://"):
                    try:
                        ws = websocket.create_connection(ep["url"], timeout=8, sslopt={"cert_reqs": ssl.CERT_NONE})
                        # Send initialize
                        init_msg = {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "initialize",
                            "params": {"clientInfo": {"name": "mcp-indexer", "version": "0.1.0"}},
                        }
                        ws.send(json.dumps(init_msg))
                        # Read a couple messages to clear any response
                        for _ in range(3):
                            try:
                                _ = ws.recv()
                            except Exception:
                                break
                        # Ask for tools/list
                        tools_msg = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
                        ws.send(json.dumps(tools_msg))
                        tools_found: List[Dict[str, str]] = []
                        deadline = time.time() + 8
                        while time.time() < deadline:
                            try:
                                raw = ws.recv()
                            except Exception:
                                break
                            try:
                                obj = json.loads(raw)
                            except Exception:
                                continue
                            if obj.get("id") == 2 and "result" in obj:
                                result = obj.get("result") or {}
                                for t in result.get("tools", []) or []:
                                    nm = t.get("name") or ""
                                    ds = t.get("description") or ""
                                    if nm:
                                        tools_found.append({"name": nm, "description": ds, "language": "n/a"})
                                break
                        ws.close()
                        if tools_found:
                            ep["verified"] = True
                            ep["verified_tools"] = tools_found
                        else:
                            ep["verified"] = False
                    except Exception:
                        ep["verified"] = False
        # HTTP(S) JSON-RPC verification best-effort: try POST to the exact URL
        for ep in deduped_eps:
            url = (ep.get("url") or "").lower()
            if url.startswith("http://") or url.startswith("https://"):
                try:
                    payload = json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}).encode("utf-8")
                    req = urllib.request.Request(ep["url"], data=payload, headers={"Content-Type": "application/json"}, method="POST")
                    with urllib.request.urlopen(req, timeout=8) as resp:
                        body = resp.read().decode("utf-8", errors="replace")
                        obj = json.loads(body)
                        if obj.get("id") == 3 and "result" in obj:
                            tools_found: List[Dict[str, str]] = []
                            for t in (obj.get("result") or {}).get("tools", []) or []:
                                nm = t.get("name") or ""
                                ds = t.get("description") or ""
                                if nm:
                                    tools_found.append({"name": nm, "description": ds, "language": "n/a"})
                            if tools_found:
                                ep["verified"] = True
                                ep["verified_tools"] = tools_found
                except Exception:
                    pass

    return server_hits, dedupe_tools(tool_defs), hints, deduped_eps


def extract_github_links(md: str) -> List[str]:
    links: List[str] = []
    # Basic markdown link pattern
    for m in re.finditer(r"\((https?://github\.com/[^)]+)\)", md):
        url = m.group(1)
        # Normalize to owner/repo if possible
        try:
            parsed = urllib.parse.urlparse(url)
            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 2:
                owner, repo = parts[0], parts[1]
                # Skip links to issues/wiki/releases etc. by taking first two
                links.append(f"{owner}/{repo}")
        except Exception:
            continue
    # Dedupe
    out: List[str] = []
    seen = set()
    for r in links:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def global_code_search_candidates() -> List[str]:
    candidates: List[str] = []
    queries = [
        '"@modelcontextprotocol/sdk" server.tool',
        '"modelcontextprotocol" @tool',
        '"modelcontextprotocol" "server.tool("',
    ]
    for q in queries:
        try:
            data = gh_request("/search/code", params={"q": q, "per_page": 50})
        except Exception:
            continue
        for item in data.get("items", []):
            repo = item.get("repository", {})
            full = repo.get("full_name")
            if full:
                candidates.append(full)
    # Dedupe
    out: List[str] = []
    seen = set()
    for r in candidates:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def global_repo_topic_candidates() -> List[str]:
    repos: List[str] = []
    queries = [
        'topic:mcp',
        'topic:mcp-server',
        'topic:modelcontextprotocol',
        'mcp in:topics',
    ]
    seen = set()
    for q in queries:
        try:
            data = gh_request("/search/repositories", params={"q": q, "per_page": 50})
        except Exception:
            continue
        for item in data.get("items", []) or []:
            full = item.get("full_name")
            if full and full not in seen:
                seen.add(full)
                repos.append(full)
    return repos


def npm_registry_candidates() -> List[str]:
    repos: List[str] = []
    urls = [
        "https://registry.npmjs.org/-/v1/search?text=modelcontextprotocol&size=100",
        "https://registry.npmjs.org/-/v1/search?text=mcp%20server&size=100",
    ]
    seen = set()
    for url in urls:
        try:
            with urllib.request.urlopen(url) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            continue
        for pkg in (data.get("objects") or []):
            pkg_info = (pkg.get("package") or {})
            repo = (pkg_info.get("links") or {}).get("repository") or pkg_info.get("repository") or ""
            # Normalize GitHub URLs to owner/repo
            if repo and "github.com" in repo:
                try:
                    parsed = urllib.parse.urlparse(repo)
                    parts = parsed.path.strip("/").split("/")
                    if len(parts) >= 2:
                        full = f"{parts[0]}/{parts[1]}"
                        if full not in seen:
                            seen.add(full)
                            repos.append(full)
                except Exception:
                    continue
    return repos


def discover_and_build_index(
    org: str,
    crawl_links: bool = False,
    crawl_depth: int = 1,
    search_global: bool = False,
    max_repos: int = 300,
    verify_endpoints: bool = False,
) -> Dict[str, dict]:
    index: Dict[str, dict] = {}
    # Seed queue with org repos
    seeds = [r["full_name"] for r in list_org_repos(org)]
    if search_global:
        seeds.extend(global_code_search_candidates())
        seeds.extend(global_repo_topic_candidates())
        seeds.extend(npm_registry_candidates())
    queue: List[Tuple[str, int]] = []
    seen: set[str] = set()
    for s in seeds:
        if s not in seen:
            seen.add(s)
            queue.append((s, 0))

    while queue and len(seen) <= max_repos:
        repo_full, depth = queue.pop(0)
        repo = get_repo(repo_full)
        if not repo:
            continue
        if repo.get("archived") or repo.get("disabled") or repo.get("fork"):
            continue
        print(f"Scanning {repo_full}...", file=sys.stderr)
        try:
            server_hits, tools, hints, endpoints = detect_and_extract_tools(repo, verify_endpoints=verify_endpoints)
        except Exception as e:
            print(f"  Error scanning {repo_full}: {e}", file=sys.stderr)
            continue
        if tools or hints:
            index[repo_full] = {
                "repo": repo_full,
                "html_url": repo.get("html_url"),
                "description": repo.get("description") or "",
                "language": repo.get("language") or "",
                "server_files": server_hits,
                "tools": tools,
                "hints": hints,
                "topics": get_repo_topics(repo_full),
                "depth": depth,
                "endpoints": endpoints,
            }

        # Crawl README links
        if crawl_links and depth < crawl_depth:
            readme = try_fetch_readme(repo_full, repo.get("default_branch", "main"))
            if readme:
                links = extract_github_links(readme)
                for full in links:
                    # Restrict recursion to the same org/owner
                    if not full.startswith(f"{org}/"):
                        continue
                    if full not in seen and len(seen) < max_repos:
                        seen.add(full)
                        queue.append((full, depth + 1))

    return index


def render_markdown(index: Dict[str, dict]) -> str:
    lines = ["# MCP Tools Index", ""]
    lines.append(f"Total servers detected: {len(index)}")
    lines.append("")
    for repo_full in sorted(index.keys()):
        entry = index[repo_full]
        lines.append(f"## {repo_full}")
        lines.append("")
        lines.append(f"- Repo: {entry.get('html_url')}")
        desc = entry.get("description") or ""
        if desc:
            lines.append(f"- Description: {desc}")
        lang = entry.get("language") or ""
        if lang:
            lines.append(f"- Primary language: {lang}")
        files = entry.get("server_files") or []
        if files:
            file_list = ", ".join(f["path"] for f in files)
            lines.append(f"- Detected in: {file_list}")
        hints = entry.get("hints") or []
        if hints:
            lines.append(f"- Hints: {', '.join(hints)}")
        topics = entry.get("topics") or []
        if topics:
            lines.append(f"- Topics: {', '.join(topics)}")
        depth = entry.get("depth")
        if depth is not None:
            lines.append(f"- Discovery depth: {depth}")
        lines.append("")
        lines.append("### Tools")
        for t in entry.get("tools", []):
            name = t.get("name", "<unknown>")
            desc = t.get("description", "").strip()
            lang = t.get("language", "")
            if desc:
                lines.append(f"- {name} ({lang}) - {desc}")
            else:
                lines.append(f"- {name} ({lang})")
        lines.append("")
        eps = entry.get("endpoints", [])
        if eps:
            lines.append("### Endpoints")
            for ep in eps:
                url = ep.get("url", "")
                src = ep.get("source", "")
                reach = ep.get("reachable")
                code = ep.get("http_status")
                meta = []
                if reach is True:
                    meta.append("reachable")
                elif reach is False:
                    meta.append("unreachable")
                if code:
                    meta.append(f"status {code}")
                suffix = f" ({', '.join(meta)})" if meta else ""
                lines.append(f"- {url} — from {src}{suffix}")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", default="modelcontextprotocol", help="GitHub org to scan")
    ap.add_argument("--out-json", default="mcp_tools_index.json", help="Path to write JSON index")
    ap.add_argument("--out-md", default="MCP_TOOLS_INDEX.md", help="Path to write Markdown index")
    ap.add_argument("--crawl-links", action="store_true", help="Crawl README GitHub links recursively")
    ap.add_argument("--crawl-depth", type=int, default=1, help="Depth for README link crawling")
    ap.add_argument("--search-global", action="store_true", help="Use global code search to seed discovery")
    ap.add_argument("--max-repos", type=int, default=600, help="Limit total repos scanned")
    ap.add_argument("--discover-global", action="store_true", help="Discover repos globally via GitHub topics/code")
    ap.add_argument("--discover-npm", action="store_true", help="Use npm registry search to seed discovery")
    ap.add_argument("--verify-endpoints", action="store_true", help="Attempt live verification of ws/http endpoints and fetch tools when possible")
    args = ap.parse_args()

    idx = discover_and_build_index(
        org=args.org,
        crawl_links=args.crawl_links,
        crawl_depth=args.crawl_depth,
        search_global=args.search_global or args.discover_global,
        max_repos=args.max_repos,
        verify_endpoints=args.verify_endpoints,
    )

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2, ensure_ascii=False)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(render_markdown(idx))

    print(f"Wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
