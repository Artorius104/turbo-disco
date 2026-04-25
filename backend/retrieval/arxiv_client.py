import logging
from typing import List, Dict

import feedparser
import httpx

logger = logging.getLogger(__name__)

ARXIV_ENDPOINT = "http://export.arxiv.org/api/query"


def search(query: str, max_results: int = 5, timeout: float = 6.0) -> List[Dict]:
    """Return list of {title, authors, year, link, abstract}. Empty list on failure."""
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(ARXIV_ENDPOINT, params=params)
            r.raise_for_status()
            feed = feedparser.parse(r.text)
    except Exception as e:
        logger.warning("arXiv search failed: %s", e)
        return []

    out: List[Dict] = []
    for entry in feed.entries:
        try:
            authors = ", ".join(a.get("name", "") for a in entry.get("authors", []))
            year = entry.get("published", "")[:4]
            link = entry.get("link", "")
            out.append({
                "title": entry.get("title", "").strip().replace("\n", " "),
                "authors": authors or "Unknown",
                "year": year or "n/a",
                "link": link,
                "abstract": entry.get("summary", "").strip().replace("\n", " "),
            })
        except Exception as e:
            logger.debug("arXiv entry parse skipped: %s", e)
    return out
