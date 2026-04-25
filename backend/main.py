import json
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import llm_client
from .retrieval import orchestrator
from .retrieval.supplier_lookup import enrich_materials
from .schemas import (
    ExperimentPlan,
    GeneratePlanRequest,
    LiteratureQCRequest,
    LiteratureQCResponse,
    Paper,
    ParsedHypothesis,
    ParseRequest,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ai_scientist")

app = FastAPI(title="AI Scientist MVP", version="0.1.0")
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


@app.post("/generate_plan", response_model=ExperimentPlan)
def generate_plan(req: GeneratePlanRequest):
    if not req.hypothesis.strip():
        raise HTTPException(status_code=400, detail="Empty hypothesis")
    system_prompt = llm_client.load_prompt("generate_plan.txt")
    parsed_block = (
        json.dumps(req.parsed.model_dump(), indent=2) if req.parsed else "(none)"
    )
    user_content = (
        f"HYPOTHESIS:\n{req.hypothesis}\n\n"
        f"PARSED_HYPOTHESIS:\n{parsed_block}\n\n"
        f"RELATED_PAPERS (cite these as P1, P2, ... when borrowing methods):\n"
        f"{_format_papers_for_prompt(req.papers)}\n"
    )
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

    logger.info("Enriching %d materials via Tavily", len(plan.materials))
    enriched = enrich_materials(plan.materials)
    return plan.model_copy(update={"materials": enriched})
