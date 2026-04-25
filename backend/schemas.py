from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ParseRequest(BaseModel):
    hypothesis: str


class ParsedHypothesis(BaseModel):
    intervention: str = ""
    outcome: str = ""
    mechanism: str = ""
    system: str = ""
    measurement: str = ""
    domain: str = ""
    keywords: List[str] = Field(default_factory=list)


class Paper(BaseModel):
    title: str
    authors: str
    year: str
    link: str
    abstract: str = ""
    similarity_score: float = 0.0


class LiteratureQCRequest(BaseModel):
    hypothesis: str
    parsed: Optional[ParsedHypothesis] = None


class LiteratureQCResponse(BaseModel):
    novelty: Literal["not found", "similar work exists", "exact match found"]
    papers: List[Paper]
    source: Literal["api:arxiv", "api:crossref", "local_fallback"]


class GeneratePlanRequest(BaseModel):
    hypothesis: str
    parsed: Optional[ParsedHypothesis] = None
    papers: List[Paper] = Field(default_factory=list)


class ProtocolStep(BaseModel):
    step: int
    title: str
    description: str
    duration: str = ""
    references: List[str] = Field(default_factory=list)


class Material(BaseModel):
    name: str
    supplier: str = ""
    catalog: str = ""
    quantity: str = ""
    unit_cost_usd: float = 0.0


class BudgetLineItem(BaseModel):
    category: str
    amount_usd: float


class Budget(BaseModel):
    total_usd: float
    line_items: List[BudgetLineItem]


class TimelinePhase(BaseModel):
    phase: str
    duration_weeks: float
    depends_on: List[str] = Field(default_factory=list)


class Validation(BaseModel):
    primary_metric: str
    success_criteria: str
    controls: List[str] = Field(default_factory=list)


class ExperimentPlan(BaseModel):
    protocol: List[ProtocolStep]
    materials: List[Material]
    budget: Budget
    timeline: List[TimelinePhase]
    validation: Validation
    references_used: List[str] = Field(default_factory=list)
