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


class ValidateHypothesisRequest(BaseModel):
    hypothesis: str
    parsed: Optional[ParsedHypothesis] = None


class ValidateHypothesisResponse(BaseModel):
    score: float = 0.0
    status: Literal["ok", "needs_revision"]
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    improved_hypothesis: str = ""


class Paper(BaseModel):
    title: str
    authors: str
    year: str
    link: str
    abstract: str = ""
    similarity_score: float = 0.0


class Protocol(BaseModel):
    title: str
    source: str = ""
    link: str = ""
    summary: str = ""
    link_status: Literal["ok", "unavailable", "unchecked"] = "unchecked"


class LiteratureQCRequest(BaseModel):
    hypothesis: str
    parsed: Optional[ParsedHypothesis] = None


SourceLabel = Literal[
    "api:arxiv",
    "api:europepmc",
    "api:openalex",
    "api:crossref",
    "local_fallback",
]


class LiteratureQCResponse(BaseModel):
    novelty: Literal["not found", "similar work exists", "exact match found"]
    papers: List[Paper]
    source: SourceLabel


class GeneratePlanRequest(BaseModel):
    hypothesis: str
    parsed: Optional[ParsedHypothesis] = None
    papers: List[Paper] = Field(default_factory=list)
    inline_feedback: List["FeedbackItem"] = Field(default_factory=list)
    previous_plan: Optional["ExperimentPlan"] = None


class ProtocolStep(BaseModel):
    step: int
    title: str
    description: str
    duration: str = ""
    references: List[str] = Field(default_factory=list)


MaterialCategory = Literal["Reagents", "Consumables", "Kits", "Equipment", "Animals", "Other"]
CostSource = Literal["catalog", "web", "estimated", "unknown"]


class Material(BaseModel):
    name: str
    supplier: str = ""
    catalog: str = ""
    quantity: str = ""
    unit_cost_usd: float = 0.0
    cost_display: str = "unknown"
    cost_source: CostSource = "unknown"
    category: MaterialCategory = "Other"
    url: str = ""


class BudgetLineItem(BaseModel):
    category: str
    amount_usd: float


class Budget(BaseModel):
    total_usd: float
    line_items: List[BudgetLineItem]
    notes: str = ""


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
    protocols_used: List[Protocol] = Field(default_factory=list)


class FeedbackItem(BaseModel):
    section: Literal["protocol", "materials", "budget", "timeline", "validation", "overall"]
    rating: Optional[int] = None
    correction: str = ""
    comment: str = ""


class FeedbackRequest(BaseModel):
    hypothesis: str
    parsed: Optional[ParsedHypothesis] = None
    plan: ExperimentPlan
    items: List[FeedbackItem]


class FeedbackResponse(BaseModel):
    stored: int


class ExportPDFRequest(BaseModel):
    hypothesis: str
    parsed: Optional[ParsedHypothesis] = None
    qc: Optional[LiteratureQCResponse] = None
    plan: ExperimentPlan


GeneratePlanRequest.model_rebuild()
