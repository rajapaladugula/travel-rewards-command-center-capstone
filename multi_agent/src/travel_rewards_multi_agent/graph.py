from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from .errors import AgentOutputError
from .models import (
    CandidatePlan,
    CandidateReview,
    CriticAssessment,
    EvidenceBundle,
    PlanBatch,
    WorkflowResult,
)
from .validation import (
    MINIMUM_PLAN_SCORE,
    score_candidate,
    validate_candidate,
    validate_evidence_bundle,
    validate_minimum_score,
)


class WorkflowState(TypedDict, total=False):
    mode: str
    question: str
    evidence: EvidenceBundle
    candidates: list[CandidatePlan]
    current_candidates: list[CandidatePlan]
    current_assessments: list[CriticAssessment]
    reviews: list[CandidateReview]
    feedback: list[str]
    critic_error: str
    retry_count: int
    replan: bool
    agent_error: str
    result: WorkflowResult


def _role(roles: Mapping[str, Any] | object, name: str):
    role = roles.get(name) if isinstance(roles, Mapping) else getattr(roles, name, None)
    if not callable(role):
        raise ValueError(f"Missing callable workflow role: {name}")
    return role


def _make_result(
    *,
    mode: str,
    status: str,
    answer: str,
    evidence: EvidenceBundle,
    candidates: list[CandidatePlan] | None = None,
    reviews: list[CandidateReview] | None = None,
    selected_candidate_id: str | None = None,
    retry_count: int = 0,
) -> WorkflowResult:
    return WorkflowResult(
        mode=mode,
        status=status,
        answer=answer,
        evidence=evidence,
        candidates=candidates or [],
        candidate_reviews=reviews or [],
        selected_candidate_id=selected_candidate_id,
        retry_count=retry_count,
    )


def build_graph(roles, minimum_score: int = MINIMUM_PLAN_SCORE):
    """Compile the bounded fact/plan workflow around injected specialist callables."""
    validate_minimum_score(minimum_score)
    evidence_role = _role(roles, "evidence")
    planner_role = _role(roles, "planner")
    critic_role = _role(roles, "critic")

    def gather_evidence(state: WorkflowState) -> WorkflowState:
        mode = state.get("mode")
        question = state.get("question", "")
        if mode not in {"fact", "plan"}:
            raise ValueError("Workflow mode must be 'fact' or 'plan'")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Workflow question must be non-empty")

        raw_evidence = evidence_role(question.strip())
        evidence = raw_evidence if isinstance(raw_evidence, EvidenceBundle) else EvidenceBundle.model_validate(raw_evidence)
        failures = validate_evidence_bundle(evidence)
        if failures:
            return {
                "evidence": evidence,
                "result": _make_result(
                    mode=mode,
                    status="failed",
                    answer="The evidence response could not be validated, so no answer or plan was produced.",
                    evidence=evidence,
                ),
            }

        if mode == "fact":
            if evidence.direct_answer is None:
                result = _make_result(
                    mode=mode,
                    status="abstained",
                    answer="I do not have sufficient cited evidence to answer that factual question.",
                    evidence=evidence,
                )
            else:
                result = _make_result(
                    mode=mode,
                    status="answered",
                    answer=evidence.direct_answer,
                    evidence=evidence,
                )
            return {"evidence": evidence, "result": result}

        if not evidence.citations:
            return {
                "evidence": evidence,
                "result": _make_result(
                    mode=mode,
                    status="clarification_required",
                    answer="I could not retrieve a relevant public benefit passage; please clarify the card or benefit before planning.",
                    evidence=evidence,
                ),
            }

        return {
            "evidence": evidence,
            "candidates": [],
            "reviews": [],
            "feedback": [],
            "retry_count": 0,
            "replan": False,
        }

    def generate_plans(state: WorkflowState) -> WorkflowState:
        evidence = state["evidence"]
        feedback = list(state.get("feedback", []))
        historical = state.get("candidates", [])
        historical_ids = [candidate.candidate_id for candidate in historical]
        if historical_ids:
            feedback.append(f"Use candidate IDs different from these earlier branches: {', '.join(historical_ids)}")
        try:
            raw_batch = planner_role(evidence.goal, evidence, feedback)
            batch = raw_batch if isinstance(raw_batch, PlanBatch) else PlanBatch.model_validate(raw_batch)
        except (AgentOutputError, ValidationError):
            return {"current_candidates": [], "agent_error": "Planner output was malformed."}

        current = list(batch.candidates)
        all_ids = [candidate.candidate_id for candidate in [*historical, *current]]
        if len(all_ids) != len(set(all_ids)):
            return {
                "current_candidates": current,
                "critic_error": "Planner reused a candidate ID across passes.",
            }
        return {
            "current_candidates": current,
            "candidates": [*historical, *current],
            "critic_error": "",
        }

    def critique_plans(state: WorkflowState) -> WorkflowState:
        if state.get("agent_error") or state.get("critic_error"):
            return {"current_assessments": []}

        evidence = state["evidence"]
        candidates = state["current_candidates"]
        try:
            raw_assessments = critic_role(evidence.goal, evidence, candidates)
            assessments = [
                assessment if isinstance(assessment, CriticAssessment) else CriticAssessment.model_validate(assessment)
                for assessment in raw_assessments
            ]
        except (AgentOutputError, TypeError, ValidationError):
            return {"current_assessments": [], "agent_error": "Critic output was malformed."}

        expected_ids = [candidate.candidate_id for candidate in candidates]
        assessment_ids = [assessment.candidate_id for assessment in assessments]
        if len(assessment_ids) != len(expected_ids) or len(set(assessment_ids)) != len(assessment_ids) or set(assessment_ids) != set(expected_ids):
            return {
                "current_assessments": assessments,
                "critic_error": "Critic must return exactly one assessment for each current candidate.",
            }
        return {"current_assessments": assessments, "critic_error": ""}

    def evaluate_plans(state: WorkflowState) -> WorkflowState:
        evidence = state["evidence"]
        mode = state["mode"]
        candidates = state.get("candidates", [])
        reviews = state.get("reviews", [])
        retry_count = state.get("retry_count", 0)
        critic_error = state.get("critic_error", "")
        agent_error = state.get("agent_error", "")

        if critic_error or agent_error:
            result = _make_result(
                mode=mode,
                status="failed",
                answer="A Planner or Critic response was incomplete or invalid, so no recommendation was selected.",
                evidence=evidence,
                candidates=candidates,
                reviews=reviews,
                retry_count=retry_count,
            )
            return {"result": result, "replan": False}

        assessments = state.get("current_assessments", [])
        assessment_by_id = {assessment.candidate_id: assessment for assessment in assessments}
        current_reviews: list[CandidateReview] = []
        eligible: list[tuple[CandidatePlan, CandidateReview, int]] = []

        for candidate in state.get("current_candidates", []):
            assessment = assessment_by_id[candidate.candidate_id]
            hard_failures = validate_candidate(candidate, evidence)
            review = CandidateReview(
                candidate_id=candidate.candidate_id,
                scores=assessment.scores,
                hard_failures=hard_failures,
                feedback=assessment.feedback,
            )
            current_reviews.append(review)
            score = score_candidate(review, hard_failures, minimum_score=minimum_score)
            if score is not None:
                eligible.append((candidate, review, score))

        all_reviews = [*reviews, *current_reviews]
        if eligible:
            selected, _review, _score = max(eligible, key=lambda item: item[2])
            result = _make_result(
                mode=mode,
                status="answered",
                answer="A candidate plan passed evidence and constraint checks. Review its cited steps before deciding; no action was taken.",
                evidence=evidence,
                candidates=candidates,
                reviews=all_reviews,
                selected_candidate_id=selected.candidate_id,
                retry_count=retry_count,
            )
            return {"reviews": all_reviews, "result": result, "replan": False}

        feedback: list[str] = []
        for review in current_reviews:
            notes = list(dict.fromkeys([*review.hard_failures, *review.feedback]))
            feedback.extend(f"{review.candidate_id}: {note}" for note in notes)
            if not notes:
                feedback.append(
                    f"{review.candidate_id}: total score was below the {minimum_score}-point selection threshold; "
                    "strengthen grounding and feasibility."
                )

        if retry_count < 1 and feedback:
            return {
                "reviews": all_reviews,
                "feedback": feedback,
                "retry_count": retry_count + 1,
                "replan": True,
                "result": None,
            }

        details = "; ".join(feedback) if feedback else "No candidate met the minimum verification score."
        result = _make_result(
            mode=mode,
            status="clarification_required",
            answer=f"I cannot recommend a plan from the current evidence. Please clarify or review these gaps: {details}",
            evidence=evidence,
            candidates=candidates,
            reviews=all_reviews,
            retry_count=retry_count,
        )
        return {"reviews": all_reviews, "result": result, "replan": False}

    def after_evidence(state: WorkflowState) -> str:
        return END if state.get("result") is not None else "planner"

    def after_evaluation(state: WorkflowState) -> str:
        return "planner" if state.get("replan", False) else END

    workflow = StateGraph(WorkflowState)
    workflow.add_node("evidence", gather_evidence)
    workflow.add_node("planner", generate_plans)
    workflow.add_node("critic", critique_plans)
    workflow.add_node("evaluate", evaluate_plans)
    workflow.add_edge(START, "evidence")
    workflow.add_conditional_edges("evidence", after_evidence, {"planner": "planner", END: END})
    workflow.add_edge("planner", "critic")
    workflow.add_edge("critic", "evaluate")
    workflow.add_conditional_edges("evaluate", after_evaluation, {"planner": "planner", END: END})
    return workflow.compile()
