# turbo-disco — The AI Scientist

Transform a natural-language scientific hypothesis into an operationally complete experiment plan in a few clicks.

Built for the **Fulcrum × Hack-Nation** "AI Scientist" challenge.

## What it does

1. **Parse** — extracts intervention, outcome, mechanism, system, measurement, domain, and keywords from a free-text hypothesis using `gpt-4o-mini` in JSON mode.
2. **Validate** — scores hypothesis quality (0–1) against four criteria: specific intervention, measurable outcome with a threshold, mechanistic reason, clear system/control. If the score is below 0.7, surfaces up to four issues, up to four suggestions, and a one-click rewrite.
3. **Literature QC** — searches arXiv → Europe PMC → OpenAlex → CrossRef in cascade order. Falls back to a local sentence-transformer index when all APIs are unreachable. Returns a novelty signal (`not found` / `similar work exists` / `exact match found`) and the top 3 papers.
4. **Generate Plan** — produces a full experiment plan (protocol steps, materials with pricing, budget, timeline, validation criteria) grounded in retrieved papers and protocols. Incorporates two tiers of reviewer feedback:
   - **Inline** (same session): current reviewer corrections are injected at the top of the prompt alongside the previous plan; the model must address every item explicitly.
   - **Prior** (cross-session): up to 3 past feedback rows for similar hypotheses are retrieved by embedding dot-product similarity and appended as low-priority context.
5. **Export PDF** — downloads the full plan as a printable, hyperlink-enabled PDF.
6. **Feedback loop** — reviewers rate and correct individual sections (protocol, materials, budget, timeline, validation, overall). Corrections are stored with sentence-transformer embeddings and retrieved for future similar hypotheses.

## Stack

| Layer | Technology | Role |
|---|---|---|
| **Frontend** | [Streamlit](https://streamlit.io) | Step-by-step wizard UI — hypothesis editor, plan renderer, PDF download |
| | Plotly + Pandas | Budget breakdown chart |
| **Backend** | [FastAPI](https://fastapi.tiangolo.com) + Pydantic v2 | REST API with strict request/response schemas |
| | [Uvicorn](https://www.uvicorn.org) | ASGI server |
| **LLM** | OpenAI Python SDK | `gpt-4o-mini` for parse/validate; `gpt-4o` for plan generation; JSON mode throughout |
| **Literature retrieval** | arXiv (feedparser), Europe PMC REST, OpenAlex REST, CrossRef REST | Priority cascade — first API to return results wins |
| | sentence-transformers (`all-MiniLM-L6-v2`) | Local embedding fallback over `backend/data/corpus.json` |
| **Protocol retrieval** | Europe PMC (Bio-protocol / Nature Protocols / JOVE / Cold Spring Harbor), OpenWetWare MediaWiki | Protocol search grounded in published source links |
| **Material enrichment** | Static catalog (`catalog.json`) → optional Tavily web search | Catalog-first pricing; Tavily restricted to 7 approved supplier domains |
| **PDF export** | [ReportLab](https://www.reportlab.com) | Timestamped, downloadable PDF with clickable links |
| **Feedback store** | SQLite (stdlib `sqlite3`) + sentence-transformers | Stores ratings/corrections with embeddings; retrieves by dot-product similarity |
| **HTTP client** | httpx | All outbound calls — literature APIs, Tavily, URL validation |
| **Config** | python-dotenv | `.env` file for API keys and model overrides |

## Architecture

```
Streamlit UI  ──HTTP──▶  FastAPI backend
                          ├── /parse                 gpt-4o-mini JSON mode
                          ├── /validate_hypothesis   gpt-4o-mini JSON mode
                          ├── /literature_qc         arXiv → EuropePMC → OpenAlex → CrossRef → local embedding
                          ├── /generate_plan         gpt-4o + protocols + catalog enrichment
                          │                          + inline feedback + prior reviewer notes
                          ├── /export_pdf            ReportLab
                          ├── /feedback              SQLite + embedding store
                          └── /health
```

**Feedback injection** has two layers: inline corrections from the current review session are placed first in the prompt (highest priority, always applied); past corrections for similar hypotheses are retrieved by embedding similarity (cosine dot-product, threshold 0.55, domain-boosted by +0.1) and appended at lower priority.

**Material enrichment** is catalog-first: `backend/data/catalog.json` (~50 curated reagents) covers the four sample domains. Items missing from the catalog optionally fall back to Tavily web search, but results must come from one of the seven allowed supplier domains (Thermo Fisher · Sigma-Aldrich · Promega · Qiagen · IDT · ATCC · Addgene). Any other domain is rejected and replaced with a supplier search URL.

## Quickstart

```bash
git clone <this repo>
cd turbo-disco

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# set OPENAI_API_KEY=sk-...
# (optional) set TAVILY_API_KEY for web fallback on missing catalog items
```

Run the backend (terminal 1):

```bash
uvicorn backend.main:app --reload --port 8000
```

Run the frontend (terminal 2):

```bash
streamlit run frontend/app.py
```

Open `http://localhost:8501`, pick one of the four sample hypotheses, and click **Run analysis**.

## Endpoints

| Method | Path | Input | Output |
|---|---|---|---|
| POST | `/parse` | `{ hypothesis }` | `ParsedHypothesis` — intervention, outcome, mechanism, system, measurement, domain, keywords |
| POST | `/validate_hypothesis` | `{ hypothesis, parsed? }` | score (0–1), status, issues, suggestions, improved_hypothesis |
| POST | `/literature_qc` | `{ hypothesis, parsed? }` | novelty signal, top-3 papers, source label |
| POST | `/generate_plan` | `{ hypothesis, parsed?, papers[], inline_feedback[], previous_plan? }` | `ExperimentPlan` — protocol steps, materials, budget, timeline, validation |
| POST | `/export_pdf` | `{ hypothesis, parsed?, qc?, plan }` | PDF byte stream (attachment) |
| POST | `/feedback` | `{ hypothesis, parsed?, plan, items[] }` | `{ stored: N }` |
| GET  | `/health` | — | `{ status: "ok" }` |

`literature_qc.source` values: `api:arxiv` · `api:europepmc` · `api:openalex` · `api:crossref` · `local_fallback`.

`/generate_plan` accepts `inline_feedback` (array of `{ section, rating?, correction?, comment? }`) and `previous_plan` for in-session revision. When present, feedback is injected at the top of the LLM prompt with temperature raised to 0.5 (vs. 0.3 for first-run plans) to encourage divergence.

## Configuration

| Variable | Required | Default | Notes |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | — | Used by `/parse`, `/validate_hypothesis`, `/generate_plan` |
| `OPENAI_MODEL_FAST` | no | `gpt-4o-mini` | Parse, validate, material extraction |
| `OPENAI_MODEL_PLAN` | no | `gpt-4o` | Plan generation |
| `CROSSREF_MAILTO` | no | — | Identifies your app in CrossRef + OpenAlex polite-pool User-Agent |
| `TAVILY_API_KEY` | no | — | Enables web fallback for materials missing from the catalog |
| `SUPPLIER_LOOKUP_TIMEOUT` | no | `8` | Tavily HTTP timeout in seconds |
| `SUPPLIER_LOOKUP_MAX_WORKERS` | no | `8` | Concurrent Tavily requests |

## Data sources

- **Literature**: arXiv, Europe PMC, OpenAlex, CrossRef — no API key required. Local fallback corpus at `backend/data/corpus.json`.
- **Protocols**: Europe PMC filtered to Bio-protocol / Nature Protocols / JOVE / Cold Spring Harbor Protocols; OpenWetWare MediaWiki search; curated assay→protocol map at `backend/data/protocols_map.json`.
- **Catalog**: hand-curated `backend/data/catalog.json` (~50 entries). Supplier allowlist: **Thermo Fisher · Sigma-Aldrich · Promega · Qiagen · IDT · ATCC · Addgene**.
- **Feedback**: SQLite at `backend/data/feedback.db`, auto-created on first POST to `/feedback`. Schema stores hypothesis embeddings for similarity retrieval.

## Demo verification

- **Hypothesis quality gate**: enter a deliberately weak hypothesis (e.g. *"FITC-dextran reduces inflammation in mice"* — no threshold, no mechanism). Expect an amber card listing issues and a suggested rewrite. Click "Use suggested rewrite" → run continues.
- **Strong hypothesis**: pick the *Gut Health* sample → green badge, automatic continuation.
- **Multi-source literature**: the source pill reads "arXiv | Europe PMC | OpenAlex | CrossRef" depending on which API wins. For bio queries, expect Europe PMC.
- **Protocol grounding**: the rendered plan shows a "Related protocols" panel linking to bio-protocol.org / nature.com/nprot / jove.com / openwetware.org.
- **Materials reliability**: in the *Cell Biology* plan, DMSO must show `Sigma-Aldrich · D8418 · sigmaaldrich.com/...`. No Bio-Rad / Agilent / govsci.com / eBay URLs. Items missing from the catalog show `TBD` + a supplier search URL.
- **Inline revision**: after a plan generates, expand the review panel, rate Materials 2/5 with a correction, and click "Revise plan". The new plan should address the correction explicitly; temperature is raised to 0.5 for this call.
- **PDF export**: click the download button → file opens with all sections, links clickable.
- **Cross-session feedback**: submit a correction, then re-run the same hypothesis in a new session — backend logs show `PRIOR_REVIEWER_NOTES` injected into the prompt.
- **Graceful degradation**: disconnect network → source pill switches to "Fallback · Local corpus"; validation and plan generation surface clean errors.

## Out of scope (this MVP)

- MCP runtime integration (the brief permitted but did not require it; this build uses public HTTP APIs directly).
- Authentication and per-user feedback isolation — feedback is global.
- Live supplier API integration (Thermo Fisher REST, etc.) — the catalog is static.
- Plan auto-regeneration triggered by background feedback ingestion — the user must re-run; prior reviews are picked up on the next request.
