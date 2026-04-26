import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import embedding_fallback

logger = logging.getLogger(__name__)

CATALOG_PATH = Path(__file__).parent.parent / "data" / "catalog.json"
MATCH_THRESHOLD = 0.72

_catalog: Optional[List[Dict]] = None
_alias_index: Optional[List[Dict]] = None
_alias_embeddings: Optional[np.ndarray] = None


def _load() -> None:
    global _catalog, _alias_index, _alias_embeddings
    if _catalog is not None:
        return
    _catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    # Flatten: one row per (alias OR canonical_name) → catalog index
    rows: List[Dict] = []
    for i, entry in enumerate(_catalog):
        names = list(entry.get("aliases") or [])
        cn = entry.get("canonical_name", "")
        if cn and cn.lower() not in {n.lower() for n in names}:
            names.append(cn)
        for n in names:
            rows.append({"name": n, "catalog_idx": i})

    _alias_index = rows
    model = embedding_fallback._load_model()
    _alias_embeddings = model.encode(
        [r["name"] for r in rows], normalize_embeddings=True, show_progress_bar=False
    )
    logger.info("Catalog index built: %d entries, %d alias rows", len(_catalog), len(rows))


def match(name: str) -> Optional[Dict]:
    """Return best catalog entry for `name` if cosine ≥ threshold, else None."""
    if not name or not name.strip():
        return None
    _load()
    assert _catalog is not None and _alias_embeddings is not None and _alias_index is not None

    model = embedding_fallback._load_model()
    q = model.encode([name], normalize_embeddings=True, show_progress_bar=False)[0]
    scores = _alias_embeddings @ q
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    if best_score < MATCH_THRESHOLD:
        return None
    cat_idx = _alias_index[best_idx]["catalog_idx"]
    entry = dict(_catalog[cat_idx])
    entry["_match_score"] = best_score
    return entry
