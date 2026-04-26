import logging
from typing import List, Dict, Tuple

from . import arxiv_client, crossref_client, embedding_fallback, europepmc_client, openalex_client

logger = logging.getLogger(__name__)


def _build_query(parsed: Dict | None, hypothesis: str) -> str:
    if parsed:
        bits = []
        for key in ("intervention", "outcome", "system"):
            v = parsed.get(key) or ""
            if v:
                bits.append(v)
        keywords = parsed.get("keywords") or []
        bits.extend(keywords[:5])
        if bits:
            return " ".join(bits)[:400]
    return hypothesis[:400]


def _classify_novelty(top_score: float) -> str:
    if top_score >= 0.85:
        return "exact match found"
    if top_score >= 0.65:
        return "similar work exists"
    return "not found"


def retrieve(hypothesis: str, parsed: Dict | None) -> Tuple[List[Dict], str, str]:
    """Returns (top_3_papers_with_scores, source_label, novelty)."""
    query = _build_query(parsed, hypothesis)

    sources = [
        ("api:arxiv", arxiv_client.search),
        ("api:europepmc", europepmc_client.search),
        ("api:openalex", openalex_client.search),
        ("api:crossref", crossref_client.search),
    ]

    for label, fn in sources:
        results = fn(query, max_results=8)
        if results:
            scored = embedding_fallback.score_papers(query, results)
            scored.sort(key=lambda p: p.get("similarity_score", 0.0), reverse=True)
            top = scored[:3]
            top_score = top[0].get("similarity_score", 0.0) if top else 0.0
            return top, label, _classify_novelty(top_score)

    logger.info("All live APIs returned empty; using local fallback")
    fb = embedding_fallback.search(query, max_results=3)
    top_score = fb[0].get("similarity_score", 0.0) if fb else 0.0
    return fb, "local_fallback", _classify_novelty(top_score)
