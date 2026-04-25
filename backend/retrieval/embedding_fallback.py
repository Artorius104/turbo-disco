import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

CORPUS_PATH = Path(__file__).parent.parent / "data" / "corpus.json"
MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_corpus: Optional[List[Dict]] = None
_corpus_embeddings: Optional[np.ndarray] = None


def _load_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformer model %s", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _load_corpus():
    global _corpus, _corpus_embeddings
    if _corpus is None:
        _corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
        model = _load_model()
        texts = [f"{p.get('title', '')} {p.get('abstract', '')}" for p in _corpus]
        _corpus_embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return _corpus, _corpus_embeddings


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return b @ a


def embed_query(text: str) -> np.ndarray:
    model = _load_model()
    return model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]


def score_papers(query: str, papers: List[Dict]) -> List[Dict]:
    """Attach a similarity_score to each paper dict (in place + returned)."""
    if not papers:
        return papers
    model = _load_model()
    q = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    texts = [f"{p.get('title', '')} {p.get('abstract', '')}" for p in papers]
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    scores = embs @ q
    for p, s in zip(papers, scores):
        p["similarity_score"] = float(s)
    return papers


def search(query: str, max_results: int = 3) -> List[Dict]:
    corpus, embs = _load_corpus()
    q = embed_query(query)
    scores = embs @ q
    idx = np.argsort(-scores)[:max_results]
    out: List[Dict] = []
    for i in idx:
        item = dict(corpus[int(i)])
        item["similarity_score"] = float(scores[int(i)])
        out.append(item)
    return out
