import logging
import os
from typing import List, Dict

import httpx

logger = logging.getLogger(__name__)

OPENALEX_ENDPOINT = "https://api.openalex.org/works"


def search(query: str, max_results: int = 8, timeout: float = 6.0) -> List[Dict]:
    mailto = os.getenv("CROSSREF_MAILTO", "hackathon@example.com")
    params = {
        "search": query,
        "per-page": max_results,
        "mailto": mailto,
    }
    try:
        with httpx.Client(timeout=timeout, headers={"User-Agent": f"AI-Scientist-MVP/0.2 (mailto:{mailto})"}) as client:
            r = client.get(OPENALEX_ENDPOINT, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        logger.warning("OpenAlex search failed: %s", e)
        return []

    out: List[Dict] = []
    for item in data.get("results", []):
        try:
            authorships = item.get("authorships") or []
            authors = ", ".join(
                (a.get("author") or {}).get("display_name", "") for a in authorships
            ).strip(", ")
            year = str(item.get("publication_year") or "n/a")
            doi = item.get("doi") or ""
            if doi and doi.startswith("https://"):
                link = doi
            elif doi:
                link = f"https://doi.org/{doi.replace('https://doi.org/', '')}"
            else:
                link = (item.get("primary_location") or {}).get("landing_page_url") or ""
            abstract_inv = item.get("abstract_inverted_index") or {}
            abstract = ""
            if abstract_inv:
                positions: list[tuple[int, str]] = []
                for word, idxs in abstract_inv.items():
                    for i in idxs:
                        positions.append((i, word))
                positions.sort()
                abstract = " ".join(w for _, w in positions)
            out.append({
                "title": (item.get("title") or item.get("display_name") or "").strip(),
                "authors": authors or "Unknown",
                "year": year,
                "link": link,
                "abstract": abstract,
            })
        except Exception as e:
            logger.debug("OpenAlex item parse skipped: %s", e)
    return out
