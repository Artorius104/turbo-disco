import logging
import os
from typing import List, Dict

import httpx

logger = logging.getLogger(__name__)

CROSSREF_ENDPOINT = "https://api.crossref.org/works"


def search(query: str, max_results: int = 5, timeout: float = 6.0) -> List[Dict]:
    mailto = os.getenv("CROSSREF_MAILTO", "hackathon@example.com")
    headers = {"User-Agent": f"AI-Scientist-MVP/0.1 (mailto:{mailto})"}
    params = {"query": query, "rows": max_results, "select": "title,author,issued,DOI,URL,abstract"}
    try:
        with httpx.Client(timeout=timeout, headers=headers) as client:
            r = client.get(CROSSREF_ENDPOINT, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        logger.warning("CrossRef search failed: %s", e)
        return []

    out: List[Dict] = []
    for item in data.get("message", {}).get("items", []):
        try:
            title_arr = item.get("title") or []
            title = title_arr[0] if title_arr else ""
            authors_list = item.get("author") or []
            authors = ", ".join(
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in authors_list
            )
            issued = (item.get("issued") or {}).get("date-parts") or [[None]]
            year = str(issued[0][0]) if issued and issued[0] and issued[0][0] else "n/a"
            doi = item.get("DOI", "")
            link = item.get("URL") or (f"https://doi.org/{doi}" if doi else "")
            abstract = (item.get("abstract") or "").replace("<jats:p>", "").replace("</jats:p>", "")
            out.append({
                "title": title.strip(),
                "authors": authors or "Unknown",
                "year": year,
                "link": link,
                "abstract": abstract.strip(),
            })
        except Exception as e:
            logger.debug("CrossRef item parse skipped: %s", e)
    return out
