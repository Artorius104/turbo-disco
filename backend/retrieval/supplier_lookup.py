import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import httpx

from .. import llm_client
from ..schemas import Material

logger = logging.getLogger(__name__)

_TIMEOUT = float(os.getenv("SUPPLIER_LOOKUP_TIMEOUT", "8.0"))
_MAX_WORKERS = int(os.getenv("SUPPLIER_LOOKUP_MAX_WORKERS", "8"))
_TAVILY_ENDPOINT = "https://api.tavily.com/search"

_SKIP_SUPPLIERS = {"other", ""}


def _tavily_search(query: str, api_key: str, max_results: int = 3) -> list[dict]:
    """Query Tavily and return list of {url, title, content}. Empty list on failure."""
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
    }
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            r = client.post(_TAVILY_ENDPOINT, json=payload)
            r.raise_for_status()
            return r.json().get("results", [])
    except Exception as e:
        logger.warning("Tavily search failed for %r: %s", query, e)
        return []


def _extract_with_llm(
    materials: List[Material],
    search_results: dict[str, list[dict]],
) -> dict[str, dict]:
    """
    Single gpt-4o-mini call: given Tavily snippets, extract catalog/price/url per material.
    Returns dict keyed by material name with enriched fields.
    """
    lines = []
    for m in materials:
        results = search_results.get(m.name, [])
        snippets = "\n".join(
            f"  [{r.get('url', '')}] {r.get('content', '')[:300]}" for r in results[:3]
        )
        lines.append(
            f"MATERIAL: {m.name} (supplier: {m.supplier})\n"
            + (snippets or "  (no results)")
        )
    context = "\n\n".join(lines)

    system_prompt = (
        "You are a lab supply specialist. Given web search snippets for lab materials, "
        "extract the catalog number, unit price in USD, and best product URL for each.\n"
        "Return a JSON object with an 'items' array, one entry per material in the same order:\n"
        '{"items": [{"name": "...", "catalog": "...", "unit_cost_usd": "...", "url": "..."}]}\n'
        "Rules:\n"
        '- If catalog is unclear or absent, write "TBD".\n'
        "- If price is unclear, write 0.0.\n"
        '- url must be the most specific product page found, or "" if none.\n'
        "- Output ONLY the JSON object, no prose."
    )

    try:
        data = llm_client.chat_json(
            system_prompt=system_prompt,
            user_content=context,
            model=os.getenv("OPENAI_MODEL_FAST", "gpt-4o-mini"),
            temperature=0.0,
            max_tokens=1200,
        )
    except Exception as e:
        logger.warning("LLM material extraction failed: %s", e)
        return {}

    return {
        item["name"]: {
            "catalog": item.get("catalog") or "TBD",
            "unit_cost_usd": float(item.get("unit_cost_usd") or 0.0),
            "url": item.get("url") or "",
        }
        for item in data.get("items", [])
        if isinstance(item, dict) and item.get("name")
    }


def enrich_materials(materials: List[Material]) -> List[Material]:
    """Enrich materials with real catalog, price, and URL via Tavily + LLM extraction."""
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        logger.warning("TAVILY_API_KEY not set; skipping material enrichment")
        return materials

    to_enrich = [
        m for m in materials if m.supplier.strip().lower() not in _SKIP_SUPPLIERS
    ]
    if not to_enrich:
        return materials

    # Parallel Tavily searches — strip parenthetical notes for cleaner queries
    def _clean(name: str) -> str:
        return name.split("(")[0].strip()

    queries = {
        m.name: f'"{_clean(m.name)}" {m.supplier} catalog number price'
        for m in to_enrich
    }

    search_results: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(to_enrich))) as pool:
        futures = {
            pool.submit(_tavily_search, q, api_key): name for name, q in queries.items()
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                search_results[name] = fut.result()
            except Exception:
                search_results[name] = []

    enriched_map = _extract_with_llm(to_enrich, search_results)

    return [
        Material(**{**m.model_dump(), **enriched_map.get(m.name, {})})
        for m in materials
    ]
