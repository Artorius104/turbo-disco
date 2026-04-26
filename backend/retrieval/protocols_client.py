import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

import httpx

from . import europepmc_client, embedding_fallback, url_validator

logger = logging.getLogger(__name__)

PROTOCOLS_MAP_PATH = Path(__file__).parent.parent / "data" / "protocols_map.json"
OWW_ENDPOINT = "https://openwetware.org/w/api.php"

PROTOCOL_JOURNAL_FILTER = (
    'JOURNAL:"Bio-protocol" OR JOURNAL:"Nature Protocols" '
    'OR JOURNAL:"Journal of Visualized Experiments" '
    'OR JOURNAL:"Cold Spring Harbor protocols"'
)

_static_map: Optional[List[Dict]] = None


def _load_map() -> List[Dict]:
    global _static_map
    if _static_map is None:
        _static_map = json.loads(PROTOCOLS_MAP_PATH.read_text(encoding="utf-8"))
    return _static_map


def _static_matches(query: str) -> List[Dict]:
    out: List[Dict] = []
    q = query.lower()
    for entry in _load_map():
        if any(kw in q for kw in entry.get("keywords", [])):
            out.append({
                "title": entry["title"],
                "source": entry.get("source", "static"),
                "link": entry.get("link", ""),
                "summary": entry.get("summary", ""),
                "_curated": True,  # static map is human-curated → trust the link
            })
    return out


def _openwetware_search(query: str, limit: int = 3, timeout: float = 6.0) -> List[Dict]:
    params = {
        "action": "opensearch",
        "search": query,
        "limit": limit,
        "namespace": 0,
        "format": "json",
    }
    try:
        with httpx.Client(timeout=timeout, headers={"User-Agent": "AI-Scientist-MVP/0.2"}) as client:
            r = client.get(OWW_ENDPOINT, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        logger.warning("OpenWetWare search failed: %s", e)
        return []

    if not isinstance(data, list) or len(data) < 4:
        return []
    titles = data[1] or []
    descriptions = data[2] or []
    links = data[3] or []
    out: List[Dict] = []
    for i, title in enumerate(titles):
        out.append({
            "title": title,
            "source": "OpenWetWare",
            "link": links[i] if i < len(links) else "",
            "summary": descriptions[i] if i < len(descriptions) else "",
        })
    return out


def _epmc_protocols(query: str, max_results: int = 5) -> List[Dict]:
    items = europepmc_client.search(query, max_results=max_results,
                                     extra_filter=PROTOCOL_JOURNAL_FILTER)
    out: List[Dict] = []
    for it in items:
        out.append({
            "title": it.get("title", ""),
            "source": it.get("journal") or "Europe PMC",
            "link": it.get("link", ""),
            "summary": (it.get("abstract") or "")[:400],
        })
    return out


def search_protocols(query: str, parsed: Optional[Dict] = None,
                     max_results: int = 3) -> List[Dict]:
    """Return up to `max_results` protocols, deduped by URL, scored by relevance."""
    seen: set[str] = set()
    candidates: List[Dict] = []

    static_query = query.lower()
    if parsed:
        static_query = " ".join([
            static_query,
            (parsed.get("measurement") or "").lower(),
            (parsed.get("intervention") or "").lower(),
        ])
    for p in _static_matches(static_query):
        key = p["link"] or p["title"]
        if key not in seen:
            seen.add(key)
            candidates.append(p)

    for p in _epmc_protocols(query):
        key = p["link"] or p["title"]
        if key and key not in seen:
            seen.add(key)
            candidates.append(p)

    if len(candidates) < max_results:
        for p in _openwetware_search(query):
            key = p["link"] or p["title"]
            if key and key not in seen:
                seen.add(key)
                candidates.append(p)

    if not candidates:
        return []

    scored = embedding_fallback.score_papers(
        query,
        [{"title": p["title"], "abstract": p.get("summary", "")} for p in candidates],
    )
    for p, s in zip(candidates, scored):
        p["_score"] = s.get("similarity_score", 0.0)
    candidates.sort(key=lambda p: p["_score"], reverse=True)

    top = candidates[: max_results * 2]
    # Validate only NON-curated links (static map is human-vetted; some publishers
    # — e.g. Bio-protocol — return WAF block codes to programmatic clients).
    needs_check = [p.get("link", "") for p in top if not p.get("_curated") and p.get("link")]
    statuses = url_validator.validate_many(needs_check)
    out: List[Dict] = []
    for p in top:
        link = p.get("link", "") or ""
        if p.get("_curated") and link:
            status = "ok"
        elif link:
            status = statuses.get(link, "unavailable")
            if status == "disallowed":
                link = ""
                status = "unavailable"
        else:
            status = "unavailable"
        out.append({
            "title": p["title"],
            "source": p.get("source", ""),
            "link": link if status == "ok" else "",
            "summary": p.get("summary", ""),
            "link_status": status if status in ("ok", "unavailable") else "unavailable",
        })
        if len(out) >= max_results:
            break
    return out
