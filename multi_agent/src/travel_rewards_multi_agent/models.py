from __future__ import annotations

from datetime import date, datetime
from typing import Literal
from urllib.parse import urlsplit

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.alias_generators import to_camel


class ContractModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        alias_generator=to_camel,
        populate_by_name=True,
    )


class Citation(ContractModel):
    id: str = Field(min_length=1)
    heading: str = Field(min_length=1)
    source_url: str = Field(min_length=8)
    captured_at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    excerpt: str = Field(min_length=1, validation_alias=AliasChoices("excerpt", "text"))

    @field_validator("source_url")
    @classmethod
    def require_http_source(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("citation source must be an HTTP(S) URL")
        return value

    @field_validator("captured_at")
    @classmethod
    def require_real_capture_date(cls, value: str) -> str:
        try:
            parsed = date.fromisoformat(value)
        except ValueError as error:
            raise ValueError("capture date must be a real YYYY-MM-DD date") from error
        if parsed.isoformat() != value:
            raise ValueError("capture date must use YYYY-MM-DD format")
        return value


class HouseholdSnapshot(ContractModel):
    product: str = Field(min_length=1)
    issuer: str = Field(min_length=1)
    annual_fee: float = Field(ge=0)
    benefit_name: str = Field(min_length=1)
    benefit_amount: float = Field(ge=0)
    used_amount: float = Field(ge=0)
    anniversary_date: str = Field(pattern=r"^\d{2}-\d{2}$")
    remaining_amount: float = Field(ge=0)

    @field_validator("anniversary_date")
    @classmethod
    def require_real_month_and_day(cls, value: str) -> str:
        try:
            datetime.strptime(f"2000-{value}", "%Y-%m-%d")
        except ValueError as error:
            raise ValueError("anniversary date must be a real MM-DD date") from error
        return value

    @model_validator(mode="after")
    def validate_benefit_amounts(self) -> HouseholdSnapshot:
        if self.used_amount > self.benefit_amount:
            raise ValueError("used amount cannot exceed the benefit amount")
        expected_remaining = self.benefit_amount - self.used_amount
        if abs(self.remaining_amount - expected_remaining) > 0.001:
            raise ValueError("remaining amount must equal benefit amount minus used amount")
        return self


class EvidenceFact(ContractModel):
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value: str | int | float


class EvidenceBundle(ContractModel):
    goal: str = Field(min_length=1)
    household: HouseholdSnapshot
    facts: list[EvidenceFact] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)
    direct_answer: str | None = None
    answer_evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evidence_references(self) -> EvidenceBundle:
        fact_ids = [fact.id for fact in self.facts]
        citation_ids = [citation.id for citation in self.citations]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("synthetic evidence IDs must be unique")
        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("citation IDs must be unique")

        known_ids = set(fact_ids) | set(citation_ids)
        if self.direct_answer is not None:
            if not self.direct_answer:
                raise ValueError("direct answer must be non-empty when present")
            if not self.answer_evidence_ids:
                raise ValueError("direct answer must cite at least one evidence ID")
        elif self.answer_evidence_ids:
            raise ValueError("answer evidence IDs cannot be present without a direct answer")

        if not set(self.answer_evidence_ids).issubset(known_ids):
            raise ValueError("direct answer references an unknown evidence ID")
        return self


StepKind = Literal["research", "verify", "compare", "calculate", "consider", "ask_user"]


class PlanStep(ContractModel):
    kind: StepKind
    description: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    requires_user_confirmation: bool

    @model_validator(mode="after")
    def require_evidence_for_advisory_steps(self) -> PlanStep:
        if self.kind != "ask_user" and not self.evidence_ids:
            raise ValueError("advisory plan steps must cite at least one evidence ID")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("plan step evidence IDs must be unique")
        return self


class CandidatePlan(ContractModel):
    candidate_id: str = Field(min_length=1)
    steps: list[PlanStep] = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)


class PlanBatch(ContractModel):
    candidates: list[CandidatePlan] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def require_unique_candidate_ids(self) -> PlanBatch:
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique within a planning pass")
        return self


class CandidateScores(ContractModel):
    grounding: int = Field(strict=True, ge=0, le=2)
    goal_fit: int = Field(strict=True, ge=0, le=2)
    feasibility: int = Field(strict=True, ge=0, le=2)
    clarity: int = Field(strict=True, ge=0, le=2)


class CriticAssessment(ContractModel):
    candidate_id: str = Field(min_length=1)
    scores: CandidateScores
    feedback: list[str] = Field(default_factory=list)


class CandidateReview(ContractModel):
    candidate_id: str = Field(min_length=1)
    scores: CandidateScores
    hard_failures: list[str] = Field(default_factory=list)
    feedback: list[str] = Field(default_factory=list)


WorkflowMode = Literal["fact", "plan"]
WorkflowStatus = Literal["answered", "abstained", "clarification_required", "failed"]


class WorkflowResult(ContractModel):
    mode: WorkflowMode
    status: WorkflowStatus
    answer: str = Field(min_length=1)
    evidence: EvidenceBundle
    candidates: list[CandidatePlan] = Field(default_factory=list)
    candidate_reviews: list[CandidateReview] = Field(default_factory=list)
    selected_candidate_id: str | None = None
    retry_count: int = Field(default=0, ge=0, le=1)
