"""
The shapes of data going in and out of the world model (Pydantic).

Two kinds:
- What the AI must hand back when it reads passages (`ExtractOut`). Pydantic
  checks the reply, so a malformed answer is caught before it reaches the database.
- What the API and the pipeline read back (`FindingOut`, `ClaimOut`, `RunExport`).

Numbers the AI can't find are sent as -1 and stored as "not stated" (None), because
strict JSON schemas for the model do not allow optional fields.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# better/worse: the outcome is good or bad for patients (survival, risk, response).
# higher/lower: the outcome is just a measurement (a mutation rate, expression, a count).
Direction = Literal["worse", "better", "higher", "lower", "none", "mixed"]


# -- what the AI returns -----------------------------------------------------

class FindingIn(BaseModel):
    passage: int = Field(description="The number of the passage this finding is in.")
    subject: str = Field(description="The gene, marker, subtype, treatment or patient feature, e.g. 'ESR1 expression'.")
    subject_kind: Literal["gene", "marker", "subtype", "treatment", "clinical", "other"]
    outcome: str = Field(description="What was measured, e.g. 'overall survival', 'breast cancer-specific survival'.")
    cohort: str = Field(description="The patient group or dataset, e.g. 'METABRIC ER-positive'. Empty if not stated.")
    direction: Direction = Field(description="For higher values or presence of the subject: worse or better (outcomes "
                                             "that are good or bad for patients), higher or lower (plain measurements "
                                             "such as a mutation rate), none (no association), or mixed.")
    effect_kind: Literal["HR", "OR", "RR", "other", "none"]
    effect: float = Field(description="The effect size as written, or -1 if not stated.")
    p_value: float = Field(description="The p-value as written, or -1 if not stated.")
    adjusted: bool = Field(description="True if the passage says the result was adjusted for other factors.")
    quote: str = Field(description="The exact words from the passage that state this finding. Copy them exactly.")


class ExtractOut(BaseModel):
    findings: list[FindingIn]


# -- what comes back out -----------------------------------------------------

class EntityOut(BaseModel):
    id: int
    kind: str
    name: str
    findings: int = 0


class FindingOut(BaseModel):
    id: int
    pmid: str
    title: str
    subject: str
    outcome: str
    cohort: str
    direction: Direction
    effect_kind: str
    effect: Optional[float]
    p_value: Optional[float]
    adjusted: bool
    quote: str

    def line(self) -> str:
        """One plain line for an AI prompt."""
        size = f", {self.effect_kind} {self.effect:g}" if self.effect is not None and self.effect_kind != "none" else ""
        p = f", p={self.p_value:g}" if self.p_value is not None else ""
        adj = ", adjusted" if self.adjusted else ""
        where = f" in {self.cohort}" if self.cohort else ""
        return (f"[PMID {self.pmid}] higher/present {self.subject} -> {self.direction} {self.outcome}"
                f"{size}{p}{adj}{where}. Quote: \"{self.quote}\"")


class LinkOut(BaseModel):
    relation: Literal["agrees", "disagrees", "related"]
    finding: FindingOut


class EvidenceOut(BaseModel):
    referee: str
    outcome: str
    detail: str
    data: dict


class VerdictOut(BaseModel):
    label: str
    triage: str
    reasons: list[str]
    rule_version: str


class ClaimOut(BaseModel):
    claim_id: str
    run_id: str
    type: str
    text: str
    evidence: list[EvidenceOut]
    verdict: Optional[VerdictOut]
    findings: list[LinkOut]


class RunExport(BaseModel):
    run_id: str
    question: str
    claims: list[ClaimOut]
    queries: Optional[int] = Field(default=None, description="SQL queries used to build this export, when counted.")


class Stats(BaseModel):
    papers: int
    passages: int
    passages_read: int
    entities: int
    findings: int
    runs: int
    claims: int
    links: int


class GraphNode(BaseModel):
    id: str
    kind: str  # entity kind, "paper" or "claim"
    label: str


class GraphEdge(BaseModel):
    source: str
    target: str
    kind: str  # finding direction, "reports" or a claim relation
    finding_id: Optional[int] = None


class Graph(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
