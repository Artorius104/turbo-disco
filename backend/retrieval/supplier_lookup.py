import logging
import os
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import httpx

from .. import llm_client
from ..schemas import Material
from . import catalog_lookup

logger = logging.getLogger(__name__)

_TIMEOUT = float(os.getenv("SUPPLIER_LOOKUP_TIMEOUT", "8.0"))
_MAX_WORKERS = int(os.getenv("SUPPLIER_LOOKUP_MAX_WORKERS", "8"))
_TAVILY_ENDPOINT = "https://api.tavily.com/search"

ALLOWED_SUPPLIERS = {
    "thermo fisher": "Thermo Fisher",
    "thermofisher": "Thermo Fisher",
    "thermo": "Thermo Fisher",
    "sigma-aldrich": "Sigma-Aldrich",
    "sigma aldrich": "Sigma-Aldrich",
    "sigma": "Sigma-Aldrich",
    "millipore sigma": "Sigma-Aldrich",
    "merck": "Sigma-Aldrich",
    "promega": "Promega",
    "qiagen": "Qiagen",
    "idt": "IDT",
    "idtdna": "IDT",
    "integrated dna technologies": "IDT",
    "atcc": "ATCC",
    "addgene": "Addgene",
}

ALLOWED_DOMAINS = {
    "sigmaaldrich.com": "Sigma-Aldrich",
    "merckmillipore.com": "Sigma-Aldrich",
    "thermofisher.com": "Thermo Fisher",
    "fishersci.com": "Thermo Fisher",
    "promega.com": "Promega",
    "qiagen.com": "Qiagen",
    "idtdna.com": "IDT",
    "atcc.org": "ATCC",
    "addgene.org": "Addgene",
}

SUPPLIER_SEARCH_URL = {
    "Sigma-Aldrich": "https://www.sigmaaldrich.com/US/en/search/{q}",
    "Thermo Fisher": "https://www.thermofisher.com/search/results?query={q}",
    "Promega": "https://www.promega.com/search?searchString={q}",
    "Qiagen": "https://www.qiagen.com/us/search?q={q}",
    "IDT": "https://www.idtdna.com/pages/search?q={q}",
    "ATCC": "https://www.atcc.org/search#q={q}",
    "Addgene": "https://www.addgene.org/search/?q={q}",
}

_SKIP_SUPPLIERS = {"other", ""}


def _normalize_supplier(value: str) -> Optional[str]:
    if not value:
        return None
    return ALLOWED_SUPPLIERS.get(value.strip().lower())


def _domain_of(url: str) -> str:
    try:
        host = urllib.parse.urlparse(url).hostname or ""
    except Exception:
        return ""
    parts = host.lower().split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host.lower()


def _supplier_for_url(url: str) -> Optional[str]:
    return ALLOWED_DOMAINS.get(_domain_of(url))


def _supplier_search_url(supplier: str, name: str) -> str:
    template = SUPPLIER_SEARCH_URL.get(supplier)
    if not template:
        template = SUPPLIER_SEARCH_URL["Sigma-Aldrich"]
    return template.format(q=urllib.parse.quote(name))


def _apply_catalog_match(m: Material) -> Optional[Material]:
    hit = catalog_lookup.match(m.name)
    if not hit:
        return None
    return Material(
        name=hit.get("canonical_name") or m.name,
        supplier=hit.get("supplier", m.supplier),
        catalog=hit.get("catalog", "TBD"),
        quantity=m.quantity or hit.get("package_size", ""),
        unit_cost_usd=float(hit.get("unit_cost_usd") or 0.0),
        url=hit.get("url", ""),
    )


def _tavily_search(query: str, api_key: str, max_results: int = 4) -> list[dict]:
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_domains": list(ALLOWED_DOMAINS.keys()),
    }
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            r = client.post(_TAVILY_ENDPOINT, json=payload)
            r.raise_for_status()
            return r.json().get("results", [])
    except Exception as e:
        logger.warning("Tavily search failed for %r: %s", query, e)
        return []


def _filter_to_allowed(results: list[dict]) -> list[dict]:
    return [r for r in results if _supplier_for_url(r.get("url", ""))]


def _extract_with_llm(materials: List[Material],
                      search_results: dict[str, list[dict]]) -> dict[str, dict]:
    lines = []
    for m in materials:
        results = search_results.get(m.name, [])
        snippets = "\n".join(
            f"  [{r.get('url', '')}] {r.get('content', '')[:300]}"
            for r in results[:3]
        )
        lines.append(
            f"MATERIAL: {m.name} (supplier hint: {m.supplier})\n"
            + (snippets or "  (no allowed-supplier results)")
        )
    context = "\n\n".join(lines)

    system_prompt = (
        "You are a lab supply specialist. Given web search snippets restricted to "
        "Thermo Fisher / Sigma-Aldrich / Promega / Qiagen / IDT / ATCC / Addgene, "
        "extract the catalog number, unit price in USD, and product URL for each material.\n"
        "Return a JSON object with an 'items' array, one entry per material in the same order:\n"
        '{"items": [{"name": "...", "supplier": "...", "catalog": "...", '
        '"unit_cost_usd": 0.0, "url": "..."}]}\n'
        "Rules:\n"
        '- supplier must be one of: "Thermo Fisher", "Sigma-Aldrich", "Promega", '
        '"Qiagen", "IDT", "ATCC", "Addgene". If the snippet does not match one of '
        'these, set supplier to "" and catalog to "TBD".\n'
        '- url must come from the snippets and host one of: sigmaaldrich.com, '
        'thermofisher.com, fishersci.com, promega.com, qiagen.com, idtdna.com, '
        'atcc.org, addgene.org. Otherwise leave url "".\n'
        '- If catalog is unclear, write "TBD". If price is unclear, write 0.0.\n'
        "- Output ONLY the JSON object."
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

    out: dict[str, dict] = {}
    for item in data.get("items", []):
        if not isinstance(item, dict) or not item.get("name"):
            continue
        url = item.get("url") or ""
        supplier = _normalize_supplier(item.get("supplier", "")) or _supplier_for_url(url)
        if not supplier:
            continue
        if url and not _supplier_for_url(url):
            url = ""
        out[item["name"]] = {
            "supplier": supplier,
            "catalog": item.get("catalog") or "TBD",
            "unit_cost_usd": float(item.get("unit_cost_usd") or 0.0),
            "url": url,
        }
    return out


def _final_fallback(m: Material) -> Material:
    supplier = _normalize_supplier(m.supplier) or "Sigma-Aldrich"
    return Material(
        name=m.name,
        supplier=supplier,
        catalog="TBD",
        quantity=m.quantity,
        unit_cost_usd=0.0,
        url=_supplier_search_url(supplier, m.name),
    )


def enrich_materials(materials: List[Material]) -> List[Material]:
    """Catalog-first material enrichment, with filtered Tavily fallback."""
    enriched: List[Material] = [None] * len(materials)  # type: ignore[list-item]
    needs_lookup: list[tuple[int, Material]] = []

    # Pass 1 — static catalog
    for i, m in enumerate(materials):
        if m.supplier.strip().lower() in _SKIP_SUPPLIERS:
            enriched[i] = _final_fallback(m)
            continue
        hit = _apply_catalog_match(m)
        if hit is not None:
            enriched[i] = hit
        else:
            needs_lookup.append((i, m))

    # Pass 2 — filtered Tavily (only if API key present)
    api_key = os.getenv("TAVILY_API_KEY", "")
    tavily_results: dict[str, list[dict]] = {}
    if api_key and needs_lookup:
        def _clean(name: str) -> str:
            return name.split("(")[0].strip()

        queries = {
            m.name: f'"{_clean(m.name)}" {m.supplier} catalog number price'
            for _, m in needs_lookup
        }
        with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(needs_lookup))) as pool:
            futures = {
                pool.submit(_tavily_search, q, api_key): name
                for name, q in queries.items()
            }
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    tavily_results[name] = _filter_to_allowed(fut.result())
                except Exception:
                    tavily_results[name] = []

        extraction = _extract_with_llm(
            [m for _, m in needs_lookup], tavily_results
        )
        still_missing: list[tuple[int, Material]] = []
        for idx, m in needs_lookup:
            data = extraction.get(m.name)
            if data and data.get("url"):
                enriched[idx] = Material(
                    name=m.name,
                    supplier=data["supplier"],
                    catalog=data["catalog"] or "TBD",
                    quantity=m.quantity,
                    unit_cost_usd=float(data.get("unit_cost_usd") or 0.0),
                    url=data["url"],
                )
            else:
                still_missing.append((idx, m))
        needs_lookup = still_missing
    else:
        if not api_key and needs_lookup:
            logger.info("TAVILY_API_KEY not set; %d materials will use search fallback URLs",
                        len(needs_lookup))

    # Pass 3 — search-URL fallback
    for idx, m in needs_lookup:
        enriched[idx] = _final_fallback(m)

    return list(enriched)
