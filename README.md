# turbo-disco — The AI Scientist

Transform a natural-language scientific hypothesis into an operationally complete experiment plan in a few clicks.

Built for the **Fulcrum × Hack-Nation** "AI Scientist" challenge.

## What it does

1. **Parse** — extracts intervention, outcome, mechanism, system, measurement, and keywords from a free-text hypothesis.
2. **Literature QC** — searches arXiv → CrossRef in order, falling back to a local sentence-transformer index over a curated corpus when APIs are unreachable. Returns a novelty signal (`not found` / `similar work exists` / `exact match found`) and the top 3 references.
3. **Generate Plan** — produces a structured experiment plan: protocol steps, materials with suppliers, budget, timeline with dependencies, and a validation strategy. Retrieved papers are injected into the prompt as grounding context.

## Architecture

```
Streamlit UI  ──HTTP──▶  FastAPI backend
                          ├── /parse           (OpenAI JSON mode)
                          ├── /literature_qc   (arXiv → CrossRef → local)
                          └── /generate_plan   (OpenAI JSON mode, papers as context)
```

Live retrieval (arXiv / CrossRef) is the primary path. If both fail or return empty, the orchestrator falls back to a local `sentence-transformers` (all-MiniLM-L6-v2) cosine search over `backend/data/corpus.json`.

## Quickstart

```bash
git clone <this repo>
cd turbo-disco

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...
```

Run the backend (terminal 1):

```bash
uvicorn backend.main:app --reload --port 8000
```

Run the frontend (terminal 2):

```bash
streamlit run frontend/app.py
```

Open the Streamlit URL (default `http://localhost:8501`), pick one of the four sample hypotheses, and click **Run analysis**.

## Endpoints

- `POST /parse` — `{ hypothesis }` → structured fields
- `POST /literature_qc` — `{ hypothesis, parsed? }` → `{ novelty, papers[], source }`
- `POST /generate_plan` — `{ hypothesis, parsed?, papers[] }` → full plan JSON
- `GET /health`

`source` is one of `api:arxiv`, `api:crossref`, `local_fallback`.

## Configuration

Environment variables (see `.env.example`):

| Variable | Required | Notes |
|---|---|---|
| `OPENAI_API_KEY` | yes | for `/parse` and `/generate_plan` |
| `OPENAI_MODEL_FAST` | no | default `gpt-4o-mini` (used for parsing) |
| `OPENAI_MODEL_PLAN` | no | default `gpt-4o` (used for plan generation) |
| `CROSSREF_MAILTO` | no | sent in User-Agent for CrossRef polite pool |

## Demo verification

- **Live data check:** with internet on, run the *Cell Biology* sample. The "Source" pill should read "Live API · …" and the top reference should have a real `doi.org` or `arxiv.org` link.
- **Graceful degradation:** disconnect Wi-Fi and run any sample. The pill should switch to "Fallback · Local corpus" and references should come from `backend/data/corpus.json`. Plan generation will fail (no LLM); the UI surfaces the error cleanly.
- **Anti-hallucination:** every protocol step that cites `P1`, `P2`, … must map to a paper in the *Top references* list above.

## Notes on MCP

The challenge brief mentions MCP tools as an option. Per the spec ("avoid over-engineering MCP integration"), this MVP uses public HTTP APIs directly. The response schema reserves the `mcp` source label for future integration.

## Out of scope (this MVP)

- Stretch goal: scientist review feedback loop / few-shot replay store.
- Authentication, persistence, multi-user.
- Caching layer (per-request latency is acceptable for a demo).
