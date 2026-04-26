import json
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from . import feedback_store, llm_client, pdf_export
from .retrieval import orchestrator, protocols_client, url_validator
from .retrieval.supplier_lookup import enrich_materials
from .schemas import (
    Budget,
    BudgetLineItem,
    ExperimentPlan,
    ExportPDFRequest,
    FeedbackRequest,
    FeedbackResponse,
    GeneratePlanRequest,
    LiteratureQCRequest,
    LiteratureQCResponse,
    Paper,
    ParsedHypothesis,
    ParseRequest,
    Protocol,
    ValidateHypothesisRequest,
    ValidateHypothesisResponse,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ai_scientist")

app = FastAPI(title="AI Scientist MVP", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/parse", response_model=ParsedHypothesis)
def parse_hypothesis(req: ParseRequest):
    if not req.hypothesis.strip():
        raise HTTPException(status_code=400, detail="Empty hypothesis")
    system_prompt = llm_client.load_prompt("parse_hypothesis.txt")
    try:
        data = llm_client.chat_json(
            system_prompt=system_prompt,
            user_content=req.hypothesis,
            model=os.getenv("OPENAI_MODEL_FAST", "gpt-4o-mini"),
            temperature=0.1,
            max_tokens=800,
        )
    except Exception as e:
        logger.exception("parse failed")
        raise HTTPException(status_code=502, detail=f"LLM parse failed: {e}")
    return ParsedHypothesis(
        intervention=data.get("intervention", "") or "",
        outcome=data.get("outcome", "") or "",
        mechanism=data.get("mechanism", "") or "",
        system=data.get("system", "") or "",
        measurement=data.get("measurement", "") or "",
        domain=data.get("domain", "") or "",
        keywords=list(data.get("keywords") or []),
    )


@app.post("/validate_hypothesis", response_model=ValidateHypothesisResponse)
def validate_hypothesis(req: ValidateHypothesisRequest):
    if not req.hypothesis.strip():
        raise HTTPException(status_code=400, detail="Empty hypothesis")
    system_prompt = llm_client.load_prompt("validate_hypothesis.txt")
    parsed_block = (
        json.dumps(req.parsed.model_dump(), indent=2) if req.parsed else "(none)"
    )
    user_content = f"HYPOTHESIS:\n{req.hypothesis}\n\nPARSED:\n{parsed_block}"
    try:
        data = llm_client.chat_json(
            system_prompt=system_prompt,
            user_content=user_content,
            model=os.getenv("OPENAI_MODEL_FAST", "gpt-4o-mini"),
            temperature=0.1,
            max_tokens=800,
        )
    except Exception as e:
        logger.exception("validate_hypothesis failed")
        raise HTTPException(status_code=502, detail=f"LLM validation failed: {e}")

    score = float(data.get("score") or 0.0)
    score = max(0.0, min(1.0, score))
    status = data.get("status") or ("ok" if score >= 0.7 else "needs_revision")
    if status not in ("ok", "needs_revision"):
        status = "ok" if score >= 0.7 else "needs_revision"
    return ValidateHypothesisResponse(
        score=score,
        status=status,  # type: ignore[arg-type]
        issues=list(data.get("issues") or [])[:4],
        suggestions=list(data.get("suggestions") or [])[:4],
        improved_hypothesis=(data.get("improved_hypothesis") or "").strip(),
    )


@app.post("/literature_qc", response_model=LiteratureQCResponse)
def literature_qc(req: LiteratureQCRequest):
    parsed_dict = req.parsed.model_dump() if req.parsed else None
    papers, source, novelty = orchestrator.retrieve(req.hypothesis, parsed_dict)
    paper_models = [
        Paper(
            title=p.get("title", "")[:500],
            authors=p.get("authors", "")[:500],
            year=str(p.get("year", "")),
            link=p.get("link", ""),
            abstract=p.get("abstract", "")[:2000],
            similarity_score=float(p.get("similarity_score", 0.0)),
        )
        for p in papers
    ]
    return LiteratureQCResponse(novelty=novelty, papers=paper_models, source=source)


import re
from pathlib import Path

_REF_TOKEN_RE = re.compile(r"^(P|PR)\d+$", re.IGNORECASE)

_RUBRIC_PATH = Path(__file__).parent / "data" / "experiment_plan_rubric.txt"
_rubric_cache: str | None = None


def _load_rubric() -> str:
    """Load the experiment-plan rubric (parsed from experiment-plan.pdf) once."""
    global _rubric_cache
    if _rubric_cache is None:
        try:
            text = _RUBRIC_PATH.read_text(encoding="utf-8").strip()
            _rubric_cache = f"RUBRIC (follow this structure and depth):\n{text}" if text else ""
        except FileNotFoundError:
            logger.info("Rubric file not found at %s — skipping injection", _RUBRIC_PATH)
            _rubric_cache = ""
    return _rubric_cache

_MATERIAL_CATEGORIES = {"Reagents", "Consumables", "Kits", "Equipment", "Animals"}
_NON_MATERIAL_CATEGORIES = {"Personnel", "Sequencing", "Other"}


def _recompute_budget(plan: ExperimentPlan) -> ExperimentPlan:
    """Compute total_usd from materials + LLM-supplied non-material line items.

    Materials with cost_source == 'unknown' are excluded from totals (price unknown);
    estimated/catalog/web prices are summed. Notes string surfaces estimated/unknown counts.
    """
    by_category: dict[str, float] = {}
    n_estimated = 0
    n_unknown = 0
    n_known = 0
    for m in plan.materials:
        if m.cost_source == "unknown":
            n_unknown += 1
            continue
        if m.cost_source == "estimated":
            n_estimated += 1
        else:
            n_known += 1
        cat = m.category if m.category in _MATERIAL_CATEGORIES else "Reagents"
        by_category[cat] = by_category.get(cat, 0.0) + float(m.unit_cost_usd or 0.0)

    materials_subtotal = sum(by_category.values())

    # Carry over LLM-supplied non-material categories (Personnel, Sequencing, Animals, Other).
    for li in plan.budget.line_items:
        cat = (li.category or "").strip()
        if cat in _NON_MATERIAL_CATEGORIES:
            amt = float(li.amount_usd or 0.0)
            # If LLM left it 0, apply a sane default tied to materials subtotal.
            if amt <= 0 and materials_subtotal > 0:
                amt = round(materials_subtotal * 0.25, -1)
            by_category[cat] = by_category.get(cat, 0.0) + amt

    # If LLM produced no Personnel line and we have a materials subtotal, add a default 1× materials.
    if "Personnel" not in by_category and materials_subtotal > 0:
        by_category["Personnel"] = round(materials_subtotal * 1.0, -1)

    line_items = [
        BudgetLineItem(category=cat, amount_usd=round(amt, 2))
        for cat, amt in sorted(by_category.items(), key=lambda kv: -kv[1])
        if amt > 0
    ]
    total = round(sum(li.amount_usd for li in line_items), 2)

    notes_parts = []
    if n_known:
        notes_parts.append(f"{n_known} priced from catalog/web")
    if n_estimated:
        notes_parts.append(f"{n_estimated} estimated by category heuristic")
    if n_unknown:
        notes_parts.append(f"{n_unknown} unknown (excluded from total)")
    notes = "; ".join(notes_parts) or "No materials priced."

    return plan.model_copy(update={
        "budget": Budget(total_usd=total, line_items=line_items, notes=notes),
    })


def _sanitize_step_refs(plan: ExperimentPlan) -> ExperimentPlan:
    """Drop disallowed/dead URLs from protocol step references; keep tokens + live URLs."""
    url_set: list[str] = []
    for step in plan.protocol:
        for r in step.references:
            if r.startswith(("http://", "https://")):
                url_set.append(r)
    statuses = url_validator.validate_many(url_set) if url_set else {}

    new_steps = []
    for step in plan.protocol:
        cleaned: list[str] = []
        for r in step.references:
            r = (r or "").strip()
            if not r:
                continue
            if _REF_TOKEN_RE.match(r):
                cleaned.append(r.upper())
                continue
            if r.startswith(("http://", "https://")):
                if statuses.get(r) == "ok":
                    cleaned.append(r)
                else:
                    cleaned.append("link unavailable")
            # else: bare text reference — drop silently
        new_steps.append(step.model_copy(update={"references": cleaned}))
    return plan.model_copy(update={"protocol": new_steps})


def _format_papers_for_prompt(papers: list[Paper]) -> str:
    if not papers:
        return "(no related papers retrieved)"
    lines = []
    for i, p in enumerate(papers, 1):
        lines.append(
            f"P{i}: {p.title}\n"
            f"  Authors: {p.authors}\n"
            f"  Year: {p.year}\n"
            f"  Link: {p.link}\n"
            f"  Abstract: {p.abstract[:600]}"
        )
    return "\n\n".join(lines)


def _format_protocols_for_prompt(protocols: list[Protocol]) -> str:
    if not protocols:
        return "(no related protocols retrieved)"
    lines = []
    for i, pr in enumerate(protocols, 1):
        lines.append(
            f"PR{i}: {pr.title}\n"
            f"  Source: {pr.source}\n"
            f"  Link: {pr.link}\n"
            f"  Summary: {pr.summary[:300]}"
        )
    return "\n\n".join(lines)


@app.post("/generate_plan", response_model=ExperimentPlan)
def generate_plan(req: GeneratePlanRequest):
    if not req.hypothesis.strip():
        raise HTTPException(status_code=400, detail="Empty hypothesis")
    system_prompt = llm_client.load_prompt("generate_plan.txt")
    parsed_dict = req.parsed.model_dump() if req.parsed else {}
    parsed_block = json.dumps(parsed_dict, indent=2) if parsed_dict else "(none)"

    # Protocol retrieval
    try:
        protocols_raw = protocols_client.search_protocols(req.hypothesis, parsed_dict)
    except Exception as e:
        logger.warning("protocol search failed: %s", e)
        protocols_raw = []
    protocols = [Protocol(**p) for p in protocols_raw]

    # Prior reviewer feedback (similarity-retrieved, low-priority context)
    try:
        prior_notes = feedback_store.relevant(req.hypothesis, parsed_dict, k=3)
    except Exception as e:
        logger.warning("feedback lookup failed: %s", e)
        prior_notes = []
    prior_block = feedback_store.format_for_prompt(prior_notes)

    # Current-session feedback (top-priority — placed FIRST in user content)
    inline_block = feedback_store.format_inline_feedback(
        [it.model_dump() for it in req.inline_feedback],
        req.previous_plan.model_dump() if req.previous_plan else None,
    )

    rubric_block = _load_rubric()

    user_parts: list[str] = []
    if inline_block:
        user_parts.append(inline_block)
    if rubric_block:
        user_parts.append(rubric_block)
    user_parts.extend([
        f"HYPOTHESIS:\n{req.hypothesis}",
        f"PARSED_HYPOTHESIS:\n{parsed_block}",
        f"RELATED_PAPERS (cite as P1, P2, ...):\n{_format_papers_for_prompt(req.papers)}",
        f"RELATED_PROTOCOLS (cite as PR1, PR2, ... or by URL):\n{_format_protocols_for_prompt(protocols)}",
    ])
    if prior_block:
        user_parts.append(prior_block)
    user_content = "\n\n".join(user_parts)

    # Higher temperature when revising under feedback to encourage divergence.
    temperature = 0.5 if inline_block else 0.3

    try:
        data = llm_client.chat_json(
            system_prompt=system_prompt,
            user_content=user_content,
            model=os.getenv("OPENAI_MODEL_PLAN", "gpt-4o"),
            temperature=temperature,
            max_tokens=4000,
        )
    except Exception as e:
        logger.exception("generate_plan failed")
        raise HTTPException(status_code=502, detail=f"LLM plan generation failed: {e}")

    try:
        plan = ExperimentPlan(**data)
    except Exception as e:
        logger.exception("plan schema validation failed; payload=%s", data)
        raise HTTPException(status_code=500, detail=f"Plan did not match schema: {e}")

    logger.info("Enriching %d materials (catalog → Tavily → heuristic)", len(plan.materials))
    enriched = enrich_materials(plan.materials)
    plan = plan.model_copy(update={
        "materials": enriched,
        "protocols_used": protocols,
    })
    plan = _recompute_budget(plan)
    plan = _sanitize_step_refs(plan)
    return plan


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(req: FeedbackRequest):
    parsed_dict = req.parsed.model_dump() if req.parsed else None
    plan_dict = req.plan.model_dump()
    items = [it.model_dump() for it in req.items]
    try:
        stored = feedback_store.record(req.hypothesis, parsed_dict, plan_dict, items)
    except Exception as e:
        logger.exception("feedback record failed")
        raise HTTPException(status_code=500, detail=f"feedback record failed: {e}")
    return FeedbackResponse(stored=stored)


@app.post("/export_pdf")
def export_pdf(req: ExportPDFRequest):
    try:
        pdf_bytes = pdf_export.render_pdf(req.hypothesis, req.parsed, req.qc, req.plan)
    except Exception as e:
        logger.exception("pdf export failed")
        raise HTTPException(status_code=500, detail=f"pdf export failed: {e}")
    import datetime as dt
    fname = f"experiment-plan-{dt.datetime.now().strftime('%Y%m%d-%H%M')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )
