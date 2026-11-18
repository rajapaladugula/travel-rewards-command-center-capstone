from importlib import import_module

import pytest

from travel_rewards_multi_agent.models import (
    CandidatePlan,
    CandidateScores,
    Citation,
    CriticAssessment,
    EvidenceBundle,
    EvidenceFact,
    HouseholdSnapshot,
    PlanBatch,
    PlanStep,
)
from travel_rewards_multi_agent.errors import AgentOutputError


def load_graph():
    try:
        return import_module("travel_rewards_multi_agent.graph")
    except ModuleNotFoundError:
        pytest.fail("LangGraph coordination is not implemented", pytrace=False)


def household():
    return HouseholdSnapshot(
        product="Chase Sapphire Reserve",
        issuer="Chase",
        annual_fee=795,
        benefit_name="annual travel credit",
        benefit_amount=300,
        used_amount=180,
        anniversary_date="08-15",
        remaining_amount=120,
    )


def evidence(*, direct_answer=None, answer_evidence_ids=None, include_citation=True, missing_data=None):
    citations = [Citation(
        id="travel-credit-1",
        heading="Travel Credit",
        source_url="https://creditcards.chase.com/rewards-credit-cards/sapphire/reserve",
        captured_at="2026-10-27",
        excerpt="The travel credit applies during the account anniversary year.",
    )] if include_citation else []
    facts = [EvidenceFact(id="fixture:remaining-credit", label="remaining credit", value=120)]
    return EvidenceBundle(
        goal="Plan around the synthetic travel credit.",
        household=household(),
        facts=facts,
        citations=citations,
        missing_data=missing_data or [],
        direct_answer=direct_answer,
        answer_evidence_ids=answer_evidence_ids or [],
    )


def candidate(candidate_id, evidence_ids=None, kind="verify"):
    return CandidatePlan(
        candidate_id=candidate_id,
        steps=[PlanStep(
            kind=kind,
            description="Verify the travel-credit eligibility using the dated account-anniversary terms.",
            evidence_ids=evidence_ids if evidence_ids is not None else ["travel-credit-1"],
            requires_user_confirmation=True,
        )],
        assumptions=[],
    )


def assessment(candidate_id, total=8, feedback=None):
    scores = CandidateScores(
        grounding=min(2, total),
        goal_fit=min(2, max(0, total - 2)),
        feasibility=min(2, max(0, total - 4)),
        clarity=min(2, max(0, total - 6)),
    )
    return CriticAssessment(candidate_id=candidate_id, scores=scores, feedback=feedback or [])


def make_roles(*, evidence_result=None, planner_result=None, critic_result=None, call_log=None):
    log = call_log if call_log is not None else []

    def evidence_role(question):
        log.append(("evidence", question))
        return evidence_result() if callable(evidence_result) else evidence_result

    def planner_role(goal, context, feedback):
        log.append(("planner", feedback))
        return planner_result(goal, context, feedback) if callable(planner_result) else planner_result

    def critic_role(goal, context, candidates):
        log.append(("critic", [item.candidate_id for item in candidates]))
        return critic_result(goal, context, candidates) if callable(critic_result) else critic_result

    return {"evidence": evidence_role, "planner": planner_role, "critic": critic_role}, log


def test_fact_mode_invokes_only_evidence_and_returns_its_cited_answer():
    graph_module = load_graph()
    bundle = evidence(
        direct_answer="The synthetic record shows $120 remaining.",
        answer_evidence_ids=["fixture:remaining-credit"],
    )
    roles, calls = make_roles(evidence_result=bundle)

    state = graph_module.build_graph(roles).invoke({"mode": "fact", "question": "How much credit remains?"})

    assert calls == [("evidence", "How much credit remains?")]
    assert state["result"].status == "answered"
    assert state["result"].answer == "The synthetic record shows $120 remaining."
    assert state["result"].candidate_reviews == []


def test_fact_mode_abstains_when_evidence_agent_has_no_supported_answer():
    graph_module = load_graph()
    roles, calls = make_roles(evidence_result=evidence(include_citation=False))

    state = graph_module.build_graph(roles).invoke({"mode": "fact", "question": "What is the reset rule?"})

    assert calls == [("evidence", "What is the reset rule?")]
    assert state["result"].status == "abstained"
    assert state["result"].answer


def test_plan_mode_runs_all_roles_and_selects_the_highest_passing_score():
    graph_module = load_graph()
    candidates = PlanBatch(candidates=[candidate("lower"), candidate("best")])
    roles, calls = make_roles(
        evidence_result=evidence(),
        planner_result=candidates,
        critic_result=lambda _goal, _context, plans: [
            assessment(plans[0].candidate_id, total=6),
            assessment(plans[1].candidate_id, total=8),
        ],
    )

    state = graph_module.build_graph(roles).invoke({"mode": "plan", "question": "Plan my next steps."})

    assert [item[0] for item in calls] == ["evidence", "planner", "critic"]
    assert state["result"].status == "answered"
    assert state["result"].selected_candidate_id == "best"
    assert len(state["result"].candidate_reviews) == 2
    assert state["result"].retry_count == 0


def test_plan_mode_uses_explicit_minimum_score_without_changing_default():
    graph_module = load_graph()
    candidates = PlanBatch(candidates=[candidate("six"), candidate("seven")])
    roles, calls = make_roles(
        evidence_result=evidence(),
        planner_result=candidates,
        critic_result=lambda _goal, _context, plans: [
            assessment(plans[0].candidate_id, total=6),
            assessment(plans[1].candidate_id, total=7),
        ],
    )

    result = graph_module.build_graph(roles, minimum_score=7).invoke(
        {"mode": "plan", "question": "Plan my next steps."}
    )["result"]

    assert result.status == "answered"
    assert result.selected_candidate_id == "seven"
    assert [item[0] for item in calls] == ["evidence", "planner", "critic"]


@pytest.mark.parametrize("minimum_score", [-1, 9, 6.0, True, "6"])
def test_build_graph_rejects_invalid_minimum_score(minimum_score):
    graph_module = load_graph()
    roles, _ = make_roles(evidence_result=evidence())

    with pytest.raises(ValueError, match="minimum_score"):
        graph_module.build_graph(roles, minimum_score=minimum_score)


def test_equal_candidate_scores_keep_the_original_candidate_order():
    graph_module = load_graph()
    candidates = PlanBatch(candidates=[candidate("first"), candidate("second")])
    roles, _ = make_roles(
        evidence_result=evidence(),
        planner_result=candidates,
        critic_result=lambda _goal, _context, plans: [
            assessment(plans[0].candidate_id, total=7),
            assessment(plans[1].candidate_id, total=7),
        ],
    )

    result = graph_module.build_graph(roles).invoke({"mode": "plan", "question": "Plan my next steps."})["result"]

    assert result.selected_candidate_id == "first"


def test_critic_feedback_drives_one_replan_and_is_retained_in_reviews():
    graph_module = load_graph()
    first_batch = PlanBatch(candidates=[candidate("first-pass")])
    second_batch = PlanBatch(candidates=[candidate("second-pass")])
    planner_calls = []

    def planner(_goal, _context, feedback):
        planner_calls.append(feedback)
        return first_batch if len(planner_calls) == 1 else second_batch

    def critic(_goal, _context, plans):
        if plans[0].candidate_id == "first-pass":
            return [assessment("first-pass", total=4, feedback=["Cite the source for the eligibility claim."])]
        return [assessment("second-pass", total=7)]

    roles, _ = make_roles(evidence_result=evidence(), planner_result=planner, critic_result=critic)
    result = graph_module.build_graph(roles).invoke({"mode": "plan", "question": "Plan my next steps."})["result"]

    assert len(planner_calls) == 2
    assert planner_calls[0] == []
    assert "Cite the source" in " ".join(planner_calls[1])
    assert result.retry_count == 1
    assert result.selected_candidate_id == "second-pass"
    assert [review.candidate_id for review in result.candidate_reviews] == ["first-pass", "second-pass"]


def test_all_failed_candidates_stop_after_one_retry():
    graph_module = load_graph()
    planner_calls = []

    def planner(_goal, _context, _feedback):
        planner_calls.append(True)
        return PlanBatch(candidates=[candidate(f"pass-{len(planner_calls)}")])

    def critic(_goal, _context, plans):
        return [assessment(plans[0].candidate_id, total=4, feedback=["Improve grounding."])]

    roles, _ = make_roles(evidence_result=evidence(), planner_result=planner, critic_result=critic)
    result = graph_module.build_graph(roles).invoke({"mode": "plan", "question": "Plan my next steps."})["result"]

    assert len(planner_calls) == 2
    assert result.retry_count == 1
    assert result.status == "clarification_required"
    assert result.selected_candidate_id is None


@pytest.mark.parametrize(
    "assessments",
    [
        lambda plans: [],
        lambda plans: [assessment(plans[0].candidate_id), assessment(plans[0].candidate_id)],
        lambda plans: [assessment("unknown-candidate")],
    ],
    ids=["missing", "duplicate", "unknown"],
)
def test_malformed_critic_assessment_set_fails_closed(assessments):
    graph_module = load_graph()
    plans = PlanBatch(candidates=[candidate("candidate-1")])
    roles, _ = make_roles(
        evidence_result=evidence(),
        planner_result=plans,
        critic_result=lambda _goal, _context, candidates: assessments(candidates),
    )

    result = graph_module.build_graph(roles).invoke({"mode": "plan", "question": "Plan my next steps."})["result"]

    assert result.status == "failed"
    assert result.selected_candidate_id is None


def test_candidate_with_unknown_evidence_is_rejected_and_replanned_despite_a_high_score():
    graph_module = load_graph()
    invalid_batch = PlanBatch(candidates=[candidate("unsupported", evidence_ids=["not-in-bundle"])])
    corrected_batch = PlanBatch(candidates=[candidate("corrected")])
    planner_calls = []

    def planner(_goal, _context, feedback):
        planner_calls.append(feedback)
        return invalid_batch if len(planner_calls) == 1 else corrected_batch

    roles, _ = make_roles(
        evidence_result=evidence(),
        planner_result=planner,
        critic_result=lambda _goal, _context, candidates: [assessment(candidates[0].candidate_id, total=8)],
    )

    result = graph_module.build_graph(roles).invoke({"mode": "plan", "question": "Plan my next steps."})["result"]

    assert len(planner_calls) == 2
    assert result.status == "answered"
    assert result.selected_candidate_id == "corrected"
    assert result.candidate_reviews[0].hard_failures
    assert not result.candidate_reviews[1].hard_failures


def test_deterministic_hard_failures_are_included_with_critic_feedback_for_replanning():
    graph_module = load_graph()
    unsupported = candidate("unsupported").model_copy(update={
        "assumptions": ["The issuer terms are current."],
    })
    corrected = candidate("corrected")
    planner_calls = []

    def planner(_goal, _context, feedback):
        planner_calls.append(feedback)
        return PlanBatch(candidates=[unsupported if len(planner_calls) == 1 else corrected])

    def critic(_goal, _context, plans):
        return [assessment(plans[0].candidate_id, total=8, feedback=["Make the step clearer."])]

    roles, _ = make_roles(evidence_result=evidence(), planner_result=planner, critic_result=critic)
    result = graph_module.build_graph(roles).invoke({"mode": "plan", "question": "Plan my next steps."})["result"]

    assert len(planner_calls) == 2
    assert "assumptions" in " ".join(planner_calls[1])
    assert "Make the step clearer." in " ".join(planner_calls[1])
    assert result.selected_candidate_id == "corrected"
    assert result.candidate_reviews[0].hard_failures


def test_plan_mode_requests_clarification_when_no_public_evidence_is_retrieved():
    graph_module = load_graph()
    roles, calls = make_roles(evidence_result=evidence(include_citation=False))

    result = graph_module.build_graph(roles).invoke({"mode": "plan", "question": "Plan my next steps."})["result"]

    assert calls == [("evidence", "Plan my next steps.")]
    assert result.status == "clarification_required"
    assert result.selected_candidate_id is None


def test_malformed_planner_output_fails_closed_without_leaking_output_or_calling_critic():
    graph_module = load_graph()
    roles, calls = make_roles(
        evidence_result=evidence(),
        planner_result=lambda *_args: (_ for _ in ()).throw(AgentOutputError("raw model output secret")),
    )

    result = graph_module.build_graph(roles).invoke(
        {"mode": "plan", "question": "Plan my next steps."}
    )["result"]

    assert result.status == "failed"
    assert result.selected_candidate_id is None
    assert [item[0] for item in calls] == ["evidence", "planner"]
    assert "secret" not in result.answer


def test_malformed_critic_output_fails_closed_without_leaking_output():
    graph_module = load_graph()
    roles, _ = make_roles(
        evidence_result=evidence(),
        planner_result=PlanBatch(candidates=[candidate("candidate-1")]),
        critic_result=lambda *_args: (_ for _ in ()).throw(AgentOutputError("raw critic output secret")),
    )

    result = graph_module.build_graph(roles).invoke(
        {"mode": "plan", "question": "Plan my next steps."}
    )["result"]

    assert result.status == "failed"
    assert result.selected_candidate_id is None
    assert "secret" not in result.answer
