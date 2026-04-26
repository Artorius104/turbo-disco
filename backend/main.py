import json
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from . import feedback_store, llm_client, pdf_export
from .retrieval import orchestrator, protocols_client
from .retrieval.supplier_lookup import enrich_materials
from .schemas import (
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

    # Prior reviewer feedback
    try:
        prior_notes = feedback_store.relevant(req.hypothesis, parsed_dict, k=3)
    except Exception as e:
        logger.warning("feedback lookup failed: %s", e)
        prior_notes = []
    prior_block = feedback_store.format_for_prompt(prior_notes)

    user_parts = [
        f"HYPOTHESIS:\n{req.hypothesis}",
        f"PARSED_HYPOTHESIS:\n{parsed_block}",
        f"RELATED_PAPERS (cite as P1, P2, ...):\n{_format_papers_for_prompt(req.papers)}",
        f"RELATED_PROTOCOLS (cite as PR1, PR2, ... or by URL):\n{_format_protocols_for_prompt(protocols)}",
    ]
    if prior_block:
        user_parts.append(prior_block)
    user_content = "\n\n".join(user_parts)

    try:
        data = llm_client.chat_json(
            system_prompt=system_prompt,
            user_content=user_content,
            model=os.getenv("OPENAI_MODEL_PLAN", "gpt-4o"),
            temperature=0.3,
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

    logger.info("Enriching %d materials (catalog → Tavily → fallback)", len(plan.materials))
    enriched = enrich_materials(plan.materials)
    return plan.model_copy(update={
        "materials": enriched,
        "protocols_used": protocols,
    })


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
