"""One-shot ingestion of experiment-plan.pdf into the corpus + a distilled rubric.

Idempotent: safely re-runnable. Existing internal_rubric rows are removed before re-adding.

Usage:
    python -m scripts.ingest_pdf [path-to-pdf]
"""
import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF = REPO_ROOT / "experiment-plan.pdf"
CORPUS_PATH = REPO_ROOT / "backend" / "data" / "corpus.json"
RUBRIC_PATH = REPO_ROOT / "backend" / "data" / "experiment_plan_rubric.txt"

CHUNK_SIZE = 1500
MAX_RUBRIC_CHARS = 4000


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n".join(parts)


_HEADER_RE = re.compile(
    r"^(?:[A-Z][A-Za-z0-9 /&,()\-]{2,60}|"
    r"\d+\.\s+[A-Za-z].{1,60}|"
    r"[IVX]+\.\s+[A-Za-z].{1,60})$"
)


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split into (heading, body) pairs by short title-case lines.

    Falls back to fixed-size chunking if no headers are found.
    """
    lines = [l.rstrip() for l in text.splitlines()]
    sections: list[tuple[str, list[str]]] = []
    current_title = "Introduction"
    buffer: list[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            buffer.append("")
            continue
        looks_like_header = (
            len(stripped) <= 80
            and _HEADER_RE.match(stripped)
            and (i + 1 < len(lines) and lines[i + 1].strip() == "")
        )
        if looks_like_header:
            if buffer:
                sections.append((current_title, buffer))
            current_title = stripped
            buffer = []
        else:
            buffer.append(line)
    if buffer:
        sections.append((current_title, buffer))

    if len(sections) <= 1:
        # Fallback: fixed chunks
        chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
        return [(f"Chunk {i+1}", chunk) for i, chunk in enumerate(chunks) if chunk.strip()]

    return [(title, "\n".join(body).strip()) for title, body in sections if "".join(body).strip()]


def build_rubric(full_text: str, sections: list[tuple[str, str]]) -> str:
    """Build the rubric. The supplied PDF is itself a worked example, so we
    pass the full text (truncated) as a few-shot reference rather than a
    bullet-point outline of section names."""
    cleaned = re.sub(r"\n{3,}", "\n\n", full_text).strip()
    if len(cleaned) > MAX_RUBRIC_CHARS:
        cleaned = cleaned[:MAX_RUBRIC_CHARS] + "\n...(truncated)"
    return (
        "EXAMPLE EXPERIMENT PLAN (use as a structural and depth reference; "
        "the new plan must match this level of operational detail and section "
        "coverage, but be tailored to the user's hypothesis):\n\n"
        + cleaned
    )


def update_corpus(sections: list[tuple[str, str]]) -> int:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    # Remove any prior internal_rubric entries (idempotency).
    corpus = [c for c in corpus if c.get("source") != "internal_rubric"]
    n_added = 0
    for title, body in sections:
        if not body.strip():
            continue
        corpus.append({
            "title": f"Experiment Plan Rubric — {title}",
            "authors": "Internal",
            "year": "2025",
            "link": "",
            "abstract": body[:2000],
            "source": "internal_rubric",
        })
        n_added += 1
    CORPUS_PATH.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    return n_added


def main(argv: list[str]) -> int:
    pdf_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_PDF
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    print(f"Reading {pdf_path} ...")
    text = extract_text(pdf_path)
    if not text.strip():
        print("Empty extraction — PDF may be image-only.", file=sys.stderr)
        return 2

    sections = split_into_sections(text)
    print(f"Split into {len(sections)} sections.")

    rubric = build_rubric(text, sections)
    RUBRIC_PATH.write_text(rubric, encoding="utf-8")
    print(f"Wrote rubric → {RUBRIC_PATH} ({len(rubric)} chars)")

    n = update_corpus(sections)
    print(f"Wrote {n} corpus entries → {CORPUS_PATH}")

    cache_marker = CORPUS_PATH.parent / ".corpus_cache_dirty"
    cache_marker.write_text("invalidate", encoding="utf-8")
    print(f"Marked {cache_marker} so embedding cache rebuilds on next load.")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
