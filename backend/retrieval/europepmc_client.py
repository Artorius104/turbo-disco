import logging
from typing import List, Dict

import httpx

logger = logging.getLogger(__name__)

EPMC_ENDPOINT = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def _build_link(item: Dict) -> str:
    doi = item.get("doi") or ""
    if doi:
        return f"https://doi.org/{doi}"
    pmid = item.get("pmid") or ""
    if pmid:
        return f"https://europepmc.org/article/MED/{pmid}"
    pmcid = item.get("pmcid") or ""
    if pmcid:
        return f"https://europepmc.org/article/PMC/{pmcid}"
    return ""


def search(query: str, max_results: int = 8, timeout: float = 6.0,
           extra_filter: str | None = None) -> List[Dict]:
    """Return list of {title, authors, year, link, abstract}. Empty on failure."""
    q = f"({query})"
    if extra_filter:
        q = f"{q} AND ({extra_filter})"
    params = {
        "query": q,
        "format": "json",
        "pageSize": max_results,
        "resultType": "core",
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(EPMC_ENDPOINT, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        logger.warning("Europe PMC search failed: %s", e)
        return []

    out: List[Dict] = []
    for item in data.get("resultList", {}).get("result", []):
        try:
            authors = item.get("authorString") or "Unknown"
            year = str(item.get("pubYear") or "n/a")
            link = _build_link(item)
            out.append({
                "title": (item.get("title") or "").strip(),
                "authors": authors,
                "year": year,
                "link": link,
                "abstract": (item.get("abstractText") or "").strip(),
                "journal": (item.get("journalTitle") or "").strip(),
            })
        except Exception as e:
            logger.debug("Europe PMC item parse skipped: %s", e)
    return out
