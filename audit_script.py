import re
import csv
import sys
import time
import math
import queue
import hashlib
import argparse
from dataclasses import dataclass, asdict
from urllib.parse import urlparse, urljoin, urldefrag

import requests
from bs4 import BeautifulSoup

# Optional but recommended
try:
    import pandas as pd
except ImportError:
    pd = None

STOPWORDS_EN = {
    "a","an","the","and","or","but","if","then","else","when","while","because","so","than",
    "to","of","in","on","at","by","for","from","with","without","into","over","under","between",
    "is","are","was","were","be","been","being","do","does","did","doing","have","has","had","having",
    "it","its","this","that","these","those","as","about","around","across","up","down","out","off",
    "you","your","yours","we","our","ours","they","their","theirs","he","him","his","she","her","hers",
    "i","me","my","mine","who","whom","which","what","where","why","how",
    "can","could","may","might","will","would","should","must",
    "not","no","yes",
    "best","top","vs","versus","review","reviews","guide","guides","2024","2025","2026"
}


USER_AGENT = "Mozilla/5.0 (compatible; InterlinkingAuditBot/1.0; +https://example.com/bot)"


@dataclass
class PageData:
    url: str
    status: int = 0
    final_url: str = ""
    title: str = ""
    h1: str = ""
    canonical: str = ""
    noindex: bool = False
    depth: int = 0
    folder: str = ""
    outlinks: int = 0
    inlinks: int = 0
    topic_key: str = ""
    html_hash: str = ""


def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s-]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokens_for_topic(s: str):
    s = normalize_text(s)
    toks = [t for t in re.split(r"[ \-_/]+", s) if t and t not in STOPWORDS_EN and len(t) > 2]
    return toks[:10]




def make_topic_key(title: str, h1: str, url: str) -> str:
    # Combine signals: title + h1 + slug
    parsed = urlparse(url)
    slug = parsed.path.strip("/").split("/")[-1]
    base = " ".join([title or "", h1 or "", slug or ""])
    toks = tokens_for_topic(base)
    if not toks:
        return ""
    # Stable key
    return " ".join(toks[:6])


def jaccard(a_tokens, b_tokens) -> float:
    a, b = set(a_tokens), set(b_tokens)
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def same_site(url: str, root_netloc: str) -> bool:
    try:
        return urlparse(url).netloc.lower() == root_netloc.lower()
    except Exception:
        return False


def clean_url(u: str) -> str:
    u, _frag = urldefrag(u)
    # remove trailing slash normalize
    if u.endswith("/") and len(u) > 8:
        u = u.rstrip("/")
    return u


def get_folder(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return "/"
    parts = path.split("/")
    return f"/{parts[0]}/" if parts else "/"


def fetch(session: requests.Session, url: str, timeout: int = 15):
    try:
        r = session.get(url, timeout=timeout, allow_redirects=True, headers={"User-Agent": USER_AGENT})
        return r
    except Exception:
        return None


def parse_page(html: str, base_url: str, root_netloc: str):
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.get_text(strip=True) if soup.title else "").strip()

    h1_tag = soup.find("h1")
    h1 = h1_tag.get_text(" ", strip=True).strip() if h1_tag else ""

    canonical = ""
    canon_tag = soup.find("link", rel=lambda x: x and "canonical" in x.lower())
    if canon_tag and canon_tag.get("href"):
        canonical = clean_url(urljoin(base_url, canon_tag["href"]))

    # noindex detection
    noindex = False
    robots = soup.find("meta", attrs={"name": re.compile(r"robots", re.I)})
    if robots and robots.get("content"):
        if "noindex" in robots["content"].lower():
            noindex = True

    # internal links
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("mailto:") or href.startswith("tel:") or href.startswith("javascript:"):
            continue
        full = clean_url(urljoin(base_url, href))
        if same_site(full, root_netloc):
            links.append(full)

    return title, h1, canonical, noindex, links


def crawl_site(start_url: str, max_pages: int, delay: float, timeout: int):
    start_url = clean_url(start_url)
    root = urlparse(start_url)
    root_netloc = root.netloc

    session = requests.Session()

    q = queue.Queue()
    q.put((start_url, 0))

    seen = set()
    pages = {}   # url -> PageData
    edges = []   # (from, to)

    while not q.empty() and len(pages) < max_pages:
        url, depth = q.get()
        url = clean_url(url)

        if url in seen:
            continue
        seen.add(url)

        r = fetch(session, url, timeout=timeout)
        if delay > 0:
            time.sleep(delay)

        pdata = PageData(url=url, depth=depth, folder=get_folder(url))
        if r is None:
            pages[url] = pdata
            continue

        pdata.status = int(getattr(r, "status_code", 0) or 0)
        pdata.final_url = clean_url(str(getattr(r, "url", url) or url))

        # Skip non-html
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "text/html" not in ctype:
            pages[url] = pdata
            continue

        html = r.text or ""
        pdata.html_hash = hashlib.md5(html.encode("utf-8", errors="ignore")).hexdigest()

        title, h1, canonical, noindex, out = parse_page(html, pdata.final_url, root_netloc)
        pdata.title = title
        pdata.h1 = h1
        pdata.canonical = canonical
        pdata.noindex = noindex

        # Save edges and enqueue new URLs
        out = list(dict.fromkeys(out))  # dedupe preserving order
        pdata.outlinks = len(out)

        for to_url in out:
            edges.append((url, to_url))
            if to_url not in seen and len(pages) + q.qsize() < max_pages:
                q.put((to_url, depth + 1))

        pdata.topic_key = make_topic_key(pdata.title, pdata.h1, pdata.url)

        pages[url] = pdata

    # compute inlinks
    in_count = {u: 0 for u in pages.keys()}
    for f, t in edges:
        if t in in_count:
            in_count[t] += 1
    for u, pdata in pages.items():
        pdata.inlinks = in_count.get(u, 0)

    return pages, edges


def detect_cannibalization(pages: dict, threshold: float = 0.55, min_group_size: int = 2):
    # Group by topic_key exact first
    groups = {}
    for p in pages.values():
        if not p.topic_key:
            continue
        groups.setdefault(p.topic_key, []).append(p)

    issues = []
    # 1) exact topic_key duplicates
    for k, arr in groups.items():
        if len(arr) >= min_group_size:
            # prioritize by inlinks as likely main
            arr_sorted = sorted(arr, key=lambda x: (x.inlinks, -x.depth), reverse=True)
            main = arr_sorted[0].url
            for other in arr_sorted[1:]:
                issues.append({
                    "type": "cannibalization_exact",
                    "topic_key": k,
                    "main_url": main,
                    "other_url": other.url,
                    "main_inlinks": arr_sorted[0].inlinks,
                    "other_inlinks": other.inlinks
                })

    # 2) fuzzy similarity among pages within same folder
    folder_map = {}
    for p in pages.values():
        folder_map.setdefault(p.folder, []).append(p)

    for folder, arr in folder_map.items():
        # compare only those with enough text signal
        candidates = [p for p in arr if p.title or p.h1]
        for i in range(len(candidates)):
            a = candidates[i]
            a_toks = tokens_for_topic((a.title or "") + " " + (a.h1 or ""))
            if not a_toks:
                continue
            for j in range(i + 1, len(candidates)):
                b = candidates[j]
                b_toks = tokens_for_topic((b.title or "") + " " + (b.h1 or ""))
                if not b_toks:
                    continue
                sim = jaccard(a_toks, b_toks)
                if sim >= threshold:
                    # choose main by inlinks
                    main, other = (a, b) if a.inlinks >= b.inlinks else (b, a)
                    issues.append({
                        "type": "cannibalization_fuzzy",
                        "folder": folder,
                        "similarity": round(sim, 3),
                        "main_url": main.url,
                        "other_url": other.url,
                        "main_title": main.title[:90],
                        "other_title": other.title[:90],
                    })

    return issues


def interlink_suggestions(pages: dict, edges: list, top_hubs: int = 10):
    # Hubs by inlinks + outlinks score
    arr = list(pages.values())
    for p in arr:
        pass
    arr_sorted = sorted(arr, key=lambda x: (x.inlinks * 2 + x.outlinks), reverse=True)
    hubs = arr_sorted[:top_hubs]

    # Build outgoing sets
    out_map = {}
    for f, t in edges:
        out_map.setdefault(f, set()).add(t)

    suggestions = []
    for hub in hubs:
        # Suggest that pages in same folder link to hub if they do not already
        same_folder = [p for p in arr if p.folder == hub.folder and p.url != hub.url]
        for p in same_folder:
            already = hub.url in out_map.get(p.url, set())
            if not already and p.status in (200, 0) and not p.noindex:
                suggestions.append({
                    "type": "add_link_to_hub",
                    "from_url": p.url,
                    "to_url": hub.url,
                    "hub_score": hub.inlinks * 2 + hub.outlinks,
                    "folder": hub.folder
                })

    return hubs, suggestions


def main():
    parser = argparse.ArgumentParser(description="Interlinking Map Builder + Anti-Canibalization (no AI)")
    parser.add_argument("--start", required=True, help="Start URL, ex: https://imfrancisco.com")
    parser.add_argument("--max_pages", type=int, default=500, help="Max pages to crawl")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between requests (seconds)")
    parser.add_argument("--timeout", type=int, default=15, help="Request timeout (seconds)")
    parser.add_argument("--out_prefix", default="out", help="Output prefix")
    parser.add_argument("--canni_threshold", type=float, default=0.55, help="Jaccard similarity threshold for fuzzy cannibalization")
    args = parser.parse_args()

    pages, edges = crawl_site(args.start, args.max_pages, args.delay, args.timeout)

    issues = []

    # Basic issues
    for p in pages.values():
        if p.inlinks == 0:
            issues.append({"type": "orphan_page", "url": p.url, "folder": p.folder})
        if p.outlinks == 0 and p.status == 200:
            issues.append({"type": "no_outlinks", "url": p.url, "folder": p.folder})
        if p.noindex:
            issues.append({"type": "noindex", "url": p.url, "folder": p.folder})
        if p.canonical and p.canonical != p.url:
            issues.append({"type": "canonical_mismatch", "url": p.url, "canonical": p.canonical})

    # Cannibalization
    issues.extend(detect_cannibalization(pages, threshold=args.canni_threshold))

    # Hubs and interlink suggestions
    hubs, suggestions = interlink_suggestions(pages, edges, top_hubs=12)

    # Write CSVs
    pages_path = f"{args.out_prefix}_pages.csv"
    edges_path = f"{args.out_prefix}_edges.csv"
    issues_path = f"{args.out_prefix}_issues.csv"
    clusters_path = f"{args.out_prefix}_clusters.csv"
    hubs_path = f"{args.out_prefix}_hubs.csv"
    sugg_path = f"{args.out_prefix}_suggestions.csv"

    # pages
    with open(pages_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(next(iter(pages.values()))) .keys()))
        w.writeheader()
        for p in pages.values():
            w.writerow(asdict(p))

    # edges
    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["from_url", "to_url"])
        w.writeheader()
        for fr, to in edges:
            w.writerow({"from_url": fr, "to_url": to})

    # issues
    with open(issues_path, "w", newline="", encoding="utf-8") as f:
        # dynamic fields
        fields = sorted({k for row in issues for k in row.keys()})
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in issues:
            w.writerow(row)

    # clusters by folder
    folder_counts = {}
    for p in pages.values():
        folder_counts.setdefault(p.folder, {"folder": p.folder, "pages": 0, "avg_inlinks": 0.0, "avg_outlinks": 0.0})
        folder_counts[p.folder]["pages"] += 1
        folder_counts[p.folder]["avg_inlinks"] += p.inlinks
        folder_counts[p.folder]["avg_outlinks"] += p.outlinks
    for v in folder_counts.values():
        v["avg_inlinks"] = round(v["avg_inlinks"] / max(1, v["pages"]), 2)
        v["avg_outlinks"] = round(v["avg_outlinks"] / max(1, v["pages"]), 2)

    with open(clusters_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["folder", "pages", "avg_inlinks", "avg_outlinks"])
        w.writeheader()
        for row in sorted(folder_counts.values(), key=lambda x: x["pages"], reverse=True):
            w.writerow(row)

    # hubs
    with open(hubs_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["url", "folder", "inlinks", "outlinks", "depth", "title"])
        w.writeheader()
        for h in hubs:
            w.writerow({
                "url": h.url,
                "folder": h.folder,
                "inlinks": h.inlinks,
                "outlinks": h.outlinks,
                "depth": h.depth,
                "title": h.title
            })

    # suggestions
    with open(sugg_path, "w", newline="", encoding="utf-8") as f:
        fields = ["type", "from_url", "to_url", "hub_score", "folder"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in suggestions:
            w.writerow({k: row.get(k, "") for k in fields})

    print("Done.")
    print(f"Pages: {len(pages)}")
    print(f"Edges: {len(edges)}")
    print(f"Issues: {len(issues)}")
    print("Outputs:")
    print(" -", pages_path)
    print(" -", edges_path)
    print(" -", issues_path)
    print(" -", clusters_path)
    print(" -", hubs_path)
    print(" -", sugg_path)


if __name__ == "__main__":
    main()
