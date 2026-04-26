import datetime as dt
import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

from .retrieval import embedding_fallback

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "data" / "feedback.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    domain TEXT,
    hypothesis_embedding BLOB,
    plan_json TEXT NOT NULL,
    section TEXT NOT NULL,
    rating INTEGER,
    correction TEXT,
    comment TEXT
);
CREATE INDEX IF NOT EXISTS feedback_domain_idx ON feedback(domain);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(_SCHEMA)
    return conn


def _embed(text: str) -> bytes:
    model = embedding_fallback._load_model()
    vec = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    return np.asarray(vec, dtype=np.float32).tobytes()


def record(hypothesis: str, parsed: Optional[Dict[str, Any]],
           plan: Dict[str, Any], items: List[Dict[str, Any]]) -> int:
    if not items:
        return 0
    ts = dt.datetime.utcnow().isoformat()
    domain = (parsed or {}).get("domain", "") or ""
    plan_json = json.dumps(plan, ensure_ascii=False)
    emb = _embed(hypothesis)

    rows = [
        (ts, hypothesis, domain, emb, plan_json,
         it.get("section", "overall"),
         int(it["rating"]) if it.get("rating") is not None else None,
         it.get("correction", "") or "",
         it.get("comment", "") or "")
        for it in items
        if (it.get("rating") is not None
            or (it.get("correction") or "").strip()
            or (it.get("comment") or "").strip())
    ]
    if not rows:
        return 0

    with _connect() as conn:
        conn.executemany(
            "INSERT INTO feedback "
            "(ts, hypothesis, domain, hypothesis_embedding, plan_json, "
            "section, rating, correction, comment) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    logger.info("Recorded %d feedback items (domain=%s)", len(rows), domain or "n/a")
    return len(rows)


def relevant(hypothesis: str, parsed: Optional[Dict[str, Any]] = None,
             k: int = 3, min_score: float = 0.55) -> List[Dict[str, Any]]:
    """Return up to k prior feedback rows ranked by hypothesis similarity.

    Domain match is weighted in (+0.1) but does not gate inclusion.
    """
    if not DB_PATH.exists():
        return []

    domain = (parsed or {}).get("domain", "") or ""
    q = embedding_fallback._load_model().encode(
        [hypothesis], normalize_embeddings=True, show_progress_bar=False
    )[0].astype(np.float32)

    with _connect() as conn:
        cur = conn.execute(
            "SELECT id, domain, section, rating, correction, comment, hypothesis_embedding "
            "FROM feedback "
            "WHERE (correction IS NOT NULL AND correction != '') "
            "   OR (comment IS NOT NULL AND comment != '') "
            "   OR (rating IS NOT NULL AND rating <= 3)"
        )
        rows = cur.fetchall()

    if not rows:
        return []

    scored: List[tuple[float, Dict[str, Any]]] = []
    for rid, dom, section, rating, correction, comment, emb_bytes in rows:
        if not emb_bytes:
            continue
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
        if emb.shape != q.shape:
            continue
        score = float(np.dot(emb, q))
        if domain and dom == domain:
            score += 0.1
        if score < min_score:
            continue
        scored.append((score, {
            "id": rid,
            "domain": dom,
            "section": section,
            "rating": rating,
            "correction": correction or "",
            "comment": comment or "",
            "score": score,
        }))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [item for _, item in scored[:k]]


def format_for_prompt(items: List[Dict[str, Any]]) -> str:
    if not items:
        return ""
    lines = ["PRIOR_REVIEWER_NOTES (apply these lessons; reviewers corrected past plans for similar hypotheses):"]
    for it in items:
        section = it.get("section", "overall")
        rating = it.get("rating")
        correction = (it.get("correction") or "").strip()
        comment = (it.get("comment") or "").strip()
        rating_str = f", rated {rating}/5" if rating is not None else ""
        body = correction or comment or "(no detail)"
        lines.append(f"- [section: {section}{rating_str}] {body}")
    return "\n".join(lines)
