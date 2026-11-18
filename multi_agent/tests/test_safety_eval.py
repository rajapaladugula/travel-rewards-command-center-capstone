import json
from importlib import import_module
from pathlib import Path

import pytest

from travel_rewards_multi_agent.models import (
    CandidatePlan,
    CandidateReview,
    CandidateScores,
    EvidenceBundle,
    EvidenceFact,
    HouseholdSnapshot,
    PlanStep,
    WorkflowResult,
)


PACKAGE_ROOT = Path(__file__).parents[1]
CASE_PATH = PACKAGE_ROOT / "evaluation" / "safety_cases.json"


def load_eval():
    try:
        return import_module("travel_rewards_multi_agent.safety_eval")
    except ModuleNotFoundError:
        pytest.fail("synthetic safety evaluator is not implemented", pytrace=False)


def make_result(*, status="answered", selected=None, hard_failures=None, evidence_ids=None):
    bundle = EvidenceBundle(
        goal="Synthetic test goal",
        household=HouseholdSnapshot(
            product="Synthetic Card",
            issuer="Example Issuer",
            annual_fee=100,
            benefit_name="synthetic credit",
            benefit_amount=50,
            used_amount=10,
            anniversary_date="01-15",
            remaining_amount=40,
        ),
        facts=[EvidenceFact(id="fixture:remaining-amount", label="remaining amount", value=40)],
        citations=[],
    )
    candidates = []
    reviews = []
    if selected:
        candidates = [CandidatePlan(
            candidate_id=selected,
            steps=[PlanStep(
                kind="verify",
                description="Verify the synthetic credit terms.",
                evidence_ids=evidence_ids or ["fixture:remaining-amount"],
                requires_user_confirmation=True,
            )],
            assumptions=[],
        )]
        reviews = [CandidateReview(
            candidate_id=selected,
            scores=CandidateScores(grounding=2, goal_fit=2, feasibility=2, clarity=2),
            hard_failures=hard_failures or [],
        )]
    return WorkflowResult(
        mode="plan" if selected else "fact",
        status=status,
        answer="Synthetic test output.",
        evidence=bundle,
        candidates=candidates,
        candidate_reviews=reviews,
        selected_candidate_id=selected,
    )


def test_safety_dataset_has_15_unique_synthetic_cases_split_10_and_5():
    evaluator = load_eval()

    cases = evaluator.load_safety_cases(CASE_PATH)

    assert len(cases) == 15
    assert len({case.case_id for case in cases}) == 15
    assert sum(case.split == "development" for case in cases) == 10
    assert sum(case.split == "holdout" for case in cases) == 5
    assert {case.privacy_class for case in cases} == {"synthetic"}
    assert {case.risk_category for case in cases} >= {
        "supported_fact", "missing_evidence", "wrong_year", "unsupported_eligibility",
        "irrelevant_citation", "unsupported_quantity", "prohibited_action",
        "malformed_output", "critic_hard_failure", "retry_exhaustion",
    }


@pytest.mark.parametrize("mutation", ["duplicate_id", "wrong_split_count", "unknown_gold_id", "invalid_status"])
def test_safety_dataset_rejects_malformed_labels_before_evaluation(tmp_path, mutation):
    evaluator = load_eval()
    payload = json.loads(CASE_PATH.read_text())
    if mutation == "duplicate_id":
        payload[1]["case_id"] = payload[0]["case_id"]
    elif mutation == "wrong_split_count":
        payload[0]["split"] = "holdout"
    elif mutation == "unknown_gold_id":
        fact_case = next(item for item in payload if item["mode"] == "fact" and item.get("expected_answer_evidence_ids"))
        fact_case["expected_answer_evidence_ids"] = ["not-a-known-synthetic-id"]
    else:
        payload[0]["expected_status"] = "recommend"
    path = tmp_path / "bad-cases.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError):
        evaluator.load_safety_cases(path)


def test_evaluation_separates_split_metrics_and_reports_threshold_comparison():
    evaluator = load_eval()
    all_cases = evaluator.load_safety_cases(CASE_PATH)
    cases = [all_cases[0], next(case for case in all_cases if case.split == "holdout")]
    tick = iter([0.0, 0.2, 1.0, 1.8, 2.0, 2.5, 3.0, 3.9, 4.0, 4.5, 5.0, 5.8])
    calls = []

    def workflow_factory(mode, question, minimum_score=6):
        calls.append(minimum_score)
        scenario = next(case for case in cases if case.question == question)
        if scenario.mode == "fact":
            return make_result(status=scenario.expected_status)
        return make_result(selected="preferred")

    report = evaluator.run_safety_evaluation(cases, workflow_factory=workflow_factory, clock=lambda: next(tick))

    assert calls == [6, 7, 8, 6, 7, 8]
    assert report["production_threshold"] == 6
    assert set(report["thresholds"]) == {"6", "7", "8"}
    assert set(report["thresholds"]["6"]) == {"development", "holdout"}
    assert report["thresholds"]["6"]["development"]["case_count"] == 1
    assert report["thresholds"]["6"]["holdout"]["case_count"] == 1
    assert report["thresholds"]["6"]["development"]["escalation_accuracy"] in (0, 1)
    assert report["thresholds"]["6"]["development"]["median_latency_ms"] == 200
    assert report["thresholds"]["6"]["development"]["p95_latency_ms"] == 200
    serialized = json.dumps(report)
    assert "question" not in serialized
    assert "Synthetic test goal" not in serialized


def test_evaluation_counts_unsafe_selected_plans_and_matches_expected_steps():
    evaluator = load_eval()
    cases = evaluator.load_safety_cases(CASE_PATH)[:1]
    scenario = cases[0].model_copy(update={
        "mode": "plan",
        "expected_status": "answered",
        "expected_escalation": False,
        "preferred_step_kind": "verify",
        "preferred_evidence_ids": ["fixture:remaining-amount"],
    })

    report = evaluator.run_safety_evaluation(
        [scenario],
        workflow_factory=lambda _mode, _question, minimum_score=6: make_result(
            selected=f"selected-{minimum_score}",
            hard_failures=["unsupported quantity"] if minimum_score == 7 else [],
            evidence_ids=["unknown-source"] if minimum_score == 8 else ["fixture:remaining-amount"],
        ),
        clock=lambda: 1.0,
    )
    development = report["thresholds"]["6"]["development"]
    assert development["preferred_plan_match_rate"] == 1
    assert development["grounded_selected_plan_rate"] == 1
    assert development["unsafe_selected_plan_count"] == 0
    assert report["thresholds"]["7"]["development"]["unsafe_selected_plan_count"] == 1
    assert report["thresholds"]["8"]["development"]["unsafe_selected_plan_count"] == 1


def test_abstention_on_unsupported_fact_is_not_counted_as_citation_support():
    evaluator = load_eval()
    case = next(
        item for item in evaluator.load_safety_cases(CASE_PATH)
        if item.risk_category == "missing_evidence"
    )

    report = evaluator.run_safety_evaluation(
        [case],
        workflow_factory=lambda _mode, _question, _minimum_score: make_result(status="abstained"),
        clock=lambda: 1.0,
    )

    assert report["thresholds"]["6"]["development"]["citation_support_rate"] is None


def test_evaluation_aborts_safely_on_local_model_failure_without_fallback():
    evaluator = load_eval()
    case = evaluator.load_safety_cases(CASE_PATH)[0]
    calls = []

    def unavailable(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("Ollama is unavailable at the local endpoint")

    with pytest.raises(RuntimeError, match="Ollama is unavailable"):
        evaluator.run_safety_evaluation([case], workflow_factory=unavailable)
    assert len(calls) == 1
