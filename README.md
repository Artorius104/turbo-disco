# turbo-disco — The AI Scientist

Transform a natural-language scientific hypothesis into an operationally complete experiment plan in a few clicks.

Built for the **Fulcrum × Hack-Nation** "AI Scientist" challenge.

## What it does

1. **Parse** — extracts intervention, outcome, mechanism, system, measurement, and keywords from a free-text hypothesis.
2. **Validate** — scores hypothesis quality against four criteria (specific intervention, measurable outcome with threshold, mechanistic reason, clear system / control). If weak, surfaces issues + a one-click suggested rewrite.
3. **Literature QC** — searches arXiv → Europe PMC → OpenAlex → CrossRef in order, falling back to a local sentence-transformer index over a curated corpus when APIs are unreachable. Returns a novelty signal (`not found` / `similar work exists` / `exact match found`) and the top 3 references.
4. **Generate Plan** — produces a structured experiment plan grounded in retrieved papers and protocols (Bio-protocol / Nature Protocols / JOVE / OpenWetWare). Materials are first matched against a static catalog of common reagents (Thermo Fisher / Sigma-Aldrich / Promega / Qiagen / IDT / ATCC / Addgene), with optional Tavily fallback restricted to those domains.
5. **Export PDF** — download the full plan as a printable report.
6. **Feedback loop** — reviewers can rate/correct sections; corrections for similar future hypotheses are injected into the prompt so the model improves over time.

## Architecture

```
Streamlit UI  ──HTTP──▶  FastAPI backend
                          ├── /parse                 (OpenAI JSON mode)
                          ├── /validate_hypothesis   (OpenAI JSON mode)
                          ├── /literature_qc         (arXiv → EuropePMC → OpenAlex → CrossRef → local)
                          ├── /generate_plan         (OpenAI + protocols + catalog enrichment + prior reviews)
                          ├── /export_pdf            (reportlab)
                          └── /feedback              (SQLite)
```

Material enrichment is **catalog-first**: 50+ curated reagents in `backend/data/catalog.json` cover the four sample domains. Materials missing from the catalog optionally fall back to Tavily web search, but the URL must come from one of the seven allowed supplier domains — anything else (eBay, Bio-Rad, Agilent, govsci.com, …) is rejected and a supplier search URL is used instead.

## Quickstart

```bash
git clone <this repo>
cd turbo-disco

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...
# (optional) set TAVILY_API_KEY for filtered web search fallback
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

| Method | Path | Purpose |
|---|---|---|
| POST | `/parse` | hypothesis → structured fields |
| POST | `/validate_hypothesis` | hypothesis → quality score + issues + suggested rewrite |
| POST | `/literature_qc` | hypothesis → novelty + papers + source label |
| POST | `/generate_plan` | hypothesis + papers → full plan (protocols + materials + budget + timeline + validation) |
| POST | `/export_pdf` | plan → PDF byte stream |
| POST | `/feedback` | reviewer rating + corrections → SQLite |
| GET  | `/health` | liveness |

`literature_qc.source` is one of `api:arxiv`, `api:europepmc`, `api:openalex`, `api:crossref`, `local_fallback`.

## Configuration

| Variable | Required | Notes |
|---|---|---|
| `OPENAI_API_KEY` | yes | for `/parse`, `/validate_hypothesis`, `/generate_plan` |
| `OPENAI_MODEL_FAST` | no | default `gpt-4o-mini` (parse, validate, material extraction) |
| `OPENAI_MODEL_PLAN` | no | default `gpt-4o` (plan generation) |
| `CROSSREF_MAILTO` | no | User-Agent for CrossRef + OpenAlex polite pool |
| `TAVILY_API_KEY` | no | optional fallback for materials missing from the catalog |
| `SUPPLIER_LOOKUP_TIMEOUT` | no | Tavily HTTP timeout, default 8 s |
| `SUPPLIER_LOOKUP_MAX_WORKERS` | no | Tavily parallelism, default 8 |

## Data sources

- **Literature**: arXiv, Europe PMC, OpenAlex, CrossRef (no API key required for any of these). Local fallback corpus at `backend/data/corpus.json`.
- **Protocols**: Europe PMC filtered to Bio-protocol / Nature Protocols / JOVE / Cold Spring Harbor protocols, plus OpenWetWare MediaWiki search, plus a curated assay→protocol map at `backend/data/protocols_map.json`.
- **Catalog**: hand-curated `backend/data/catalog.json` (~50 entries). Suppliers restricted to **Thermo Fisher · Sigma-Aldrich · Promega · Qiagen · IDT · ATCC · Addgene** (the brief's list).
- **Feedback**: SQLite at `backend/data/feedback.db`, auto-created on first POST to `/feedback`.

## Demo verification

- **Hypothesis quality gate**: type a deliberately weak hypothesis (e.g. *"FITC-dextran reduces inflammation in mice"* — no threshold, no mechanism). Expect an amber card listing issues and a suggested rewrite. Click "Use suggested rewrite" → the run continues.
- **Strong hypothesis**: pick the *Gut Health* sample → green badge, automatic continuation.
- **Multi-source literature**: the Source pill should read one of "Live API · arXiv | Europe PMC | OpenAlex | CrossRef" depending on which API answers first. For bio queries, expect Europe PMC.
- **Protocol grounding**: the rendered plan shows a "Related protocols" panel with at least one entry linking to bio-protocol.org / nature.com/nprot / jove.com / openwetware.org.
- **Materials reliability**: in the *Cell Biology* plan, DMSO must show `Sigma-Aldrich · D8418 · sigmaaldrich.com/...`. **No** Bio-Rad / Agilent / govsci.com / eBay URLs anywhere. Items missing from the catalog show `TBD` + a supplier search URL.
- **PDF export**: click the download button → file opens with all sections, links clickable.
- **Feedback loop**: open the review expander, rate Materials 2/5 with a correction, submit. Re-run the same hypothesis — backend logs show `PRIOR_REVIEWER_NOTES` injected into the prompt.
- **Graceful degradation**: disconnect network → source pill switches to "Fallback · Local corpus", validation/plan generation surface clean errors.

## Notes on MCP

The challenge brief mentions MCP tools as an option. Per the spec ("avoid over-engineering MCP integration"), this MVP uses public HTTP APIs directly.

## Out of scope (this MVP)

- MCP runtime integration.
- Authentication, multi-user, per-user feedback isolation (feedback is global).
- Live supplier-API integration (Thermo Fisher REST, etc.) — the catalog is static.
- Plan auto-regeneration triggered by feedback (the user must re-run; prior reviews are picked up next time).
