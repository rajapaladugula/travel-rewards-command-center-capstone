import json
from importlib import import_module
from io import StringIO
from types import SimpleNamespace

import pytest

from travel_rewards_multi_agent.graph import build_graph
from travel_rewards_multi_agent.errors import AgentOutputError
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


def load_cli():
    try:
        return import_module("travel_rewards_multi_agent.cli")
    except ModuleNotFoundError:
        pytest.fail("local multi-agent CLI is not implemented", pytrace=False)


def evidence(*, direct_answer=None, answer_evidence_ids=None, include_citation=True):
    citations = [Citation(
        id="travel-credit-1",
        heading="Travel Credit",
        source_url="https://creditcards.chase.com/rewards-credit-cards/sapphire/reserve",
        captured_at="2026-10-27",
        excerpt="The travel credit applies during the account anniversary year.",
    )] if include_citation else []
    return EvidenceBundle(
        goal="Plan around the synthetic travel credit.",
        household=HouseholdSnapshot(
            product="Chase Sapphire Reserve",
            issuer="Chase",
            annual_fee=795,
            benefit_name="annual travel credit",
            benefit_amount=300,
            used_amount=180,
            anniversary_date="08-15",
            remaining_amount=120,
        ),
        facts=[EvidenceFact(id="fixture:remaining-credit", label="remaining credit", value=120)],
        citations=citations,
        missing_data=[],
        direct_answer=direct_answer,
        answer_evidence_ids=answer_evidence_ids or [],
    )


def make_runtime(evidence_bundle, plans=None, critic=None):
    def runtime_factory(mode, question):
        roles = {
            "evidence": lambda _question: evidence_bundle,
            "planner": lambda _goal, _evidence, _feedback: plans,
            "critic": lambda _goal, _evidence, candidates: critic(candidates),
        }
        return build_graph(roles).invoke({"mode": mode, "question": question})["result"]

    return runtime_factory


def test_fact_cli_returns_cited_answer_without_running_planner_or_critic():
    cli = load_cli()
    runtime_factory = make_runtime(
        evidence(
            direct_answer="The synthetic record shows $120 remaining.",
            answer_evidence_ids=["fixture:remaining-credit"],
        ),
        plans=None,
        critic=lambda _candidates: pytest.fail("fact mode must bypass the planner and critic"),
    )
    output = StringIO()
    errors = StringIO()

    exit_code = cli.run_cli(
        ["--mode", "fact", "--question", "How much travel credit remains?"],
        runtime_factory,
        output,
        errors,
    )

    response = json.loads(output.getvalue())
    assert exit_code == 0
    assert response["status"] == "answered"
    assert response["answer"] == "The synthetic record shows $120 remaining."
    assert response["candidate_reviews"] == []
    assert response["candidates"] == []
    assert errors.getvalue() == ""


def test_plan_cli_exposes_candidate_steps_citations_and_verifier_review():
    cli = load_cli()
    plan = CandidatePlan(
        candidate_id="supported-next-step",
        steps=[PlanStep(
            kind="verify",
            description="Verify eligible travel purchases against the dated issuer passage.",
            evidence_ids=["travel-credit-1"],
            requires_user_confirmation=True,
        )],
        assumptions=[],
    )
    plans = PlanBatch(candidates=[plan])
    assessments = lambda candidates: [CriticAssessment(
        candidate_id=candidates[0].candidate_id,
        scores=CandidateScores(grounding=2, goal_fit=2, feasibility=2, clarity=2),
        feedback=["The suggestion is grounded in the cited passage."],
    )]
    output = StringIO()
    errors = StringIO()

    exit_code = cli.run_cli(
        ["--mode", "plan", "--question", "What should I verify next?"],
        make_runtime(evidence(), plans, assessments),
        output,
        errors,
    )

    response = json.loads(output.getvalue())
    assert exit_code == 0
    assert response["status"] == "answered"
    assert response["selected_candidate_id"] == "supported-next-step"
    assert response["candidates"][0]["steps"][0]["description"].startswith("Verify eligible")
    assert response["candidate_reviews"][0]["scores"]["grounding"] == 2
    assert response["evidence"]["citations"][0]["source_url"].startswith("https://")
    assert errors.getvalue() == ""


def test_fact_cli_abstains_when_retrieval_has_no_eligible_evidence():
    cli = load_cli()
    output = StringIO()

    exit_code = cli.run_cli(
        ["--mode", "fact", "--question", "What is the gasoline benefit?"],
        make_runtime(evidence(include_citation=False)),
        output,
        StringIO(),
    )

    response = json.loads(output.getvalue())
    assert exit_code == 0
    assert response["status"] == "abstained"
    assert response["evidence"]["citations"] == []


def test_cli_reports_local_model_failure_without_hosted_fallback():
    cli = load_cli()
    output = StringIO()
    errors = StringIO()

    def unavailable_runtime(_mode, _question):
        raise RuntimeError("Ollama is unavailable at the local endpoint")

    exit_code = cli.run_cli(
        ["--mode", "plan", "--question", "Plan next steps."],
        unavailable_runtime,
        output,
        errors,
    )

    assert exit_code != 0
    assert "Ollama is unavailable" in errors.getvalue()
    assert output.getvalue() == ""
    assert "hosted" not in errors.getvalue().lower()


def test_cli_hides_unexpected_exception_details_and_writes_no_partial_result():
    cli = load_cli()
    output = StringIO()
    errors = StringIO()

    def broken_runtime(_mode, _question):
        raise RuntimeError(f"secret-token /private/path {'trace ' * 100}")

    exit_code = cli.run_cli(
        ["--mode", "plan", "--question", "Plan next steps."],
        broken_runtime,
        output,
        errors,
    )

    assert exit_code != 0
    assert "secret-token" not in errors.getvalue()
    assert "/private/path" not in errors.getvalue()
    assert len(errors.getvalue()) <= 300
    assert output.getvalue() == ""


def test_task_output_parser_validates_json_text_when_model_has_no_pydantic_output():
    cli = load_cli()
    payload = {
        "goal": "Understand the remaining synthetic travel credit.",
        "household": {
            "product": "Chase Sapphire Reserve",
            "issuer": "Chase",
            "annual_fee": 795,
            "benefit_name": "annual travel credit",
            "benefit_amount": 300,
            "used_amount": 180,
            "anniversary_date": "08-15",
            "remaining_amount": 120,
        },
        "facts": [],
        "citations": [],
        "missing_data": [],
        "direct_answer": None,
        "answer_evidence_ids": [],
    }
    output = SimpleNamespace(
        tasks_output=[SimpleNamespace(pydantic=None, raw=json.dumps(payload))],
        pydantic=None,
    )

    parsed = cli._parse_task_output(output, EvidenceBundle)

    assert isinstance(parsed, EvidenceBundle)
    assert parsed.household.remaining_amount == 120


def test_task_output_parser_rejects_non_json_model_text():
    cli = load_cli()
    output = SimpleNamespace(
        tasks_output=[SimpleNamespace(pydantic=None, raw="Not valid JSON")],
        pydantic=None,
    )

    with pytest.raises(AgentOutputError, match="valid JSON"):
        cli._parse_task_output(output, EvidenceBundle)


def test_task_output_parser_normalizes_candidate_array_to_plan_batch():
    cli = load_cli()
    candidate = {
        "candidate_id": "candidate-1",
        "steps": [{
            "kind": "verify",
            "description": "Verify the issuer terms.",
            "evidence_ids": ["travel-credit-1"],
            "requires_user_confirmation": True,
        }],
        "assumptions": [],
    }
    output = SimpleNamespace(
        tasks_output=[SimpleNamespace(pydantic=None, raw=json.dumps([candidate]))],
        pydantic=None,
    )

    parsed = cli._parse_task_output(output, PlanBatch)

    assert isinstance(parsed, PlanBatch)
    assert parsed.candidates[0].candidate_id == "candidate-1"


def test_task_output_parser_promotes_step_assumptions_to_candidate_level():
    cli = load_cli()
    candidate = {
        "candidate_id": "candidate-1",
        "steps": [{
            "kind": "verify",
            "description": "Verify the issuer terms.",
            "evidence_ids": ["travel-credit-1"],
            "requires_user_confirmation": True,
            "assumptions": ["The public snapshot remains applicable."],
        }],
    }
    output = SimpleNamespace(
        tasks_output=[SimpleNamespace(pydantic=None, raw=json.dumps({"candidates": [candidate]}))],
        pydantic=None,
    )

    parsed = cli._parse_task_output(output, PlanBatch)

    assert parsed.candidates[0].assumptions == ["The public snapshot remains applicable."]


def test_planner_task_specifies_candidate_and_step_json_fields(monkeypatch):
    cli = load_cli()
    captured = {}

    def capture_task(_agent, task, _model, inputs):
        captured["task"] = task
        captured["inputs"] = inputs
        return PlanBatch(candidates=[CandidatePlan(
            candidate_id="candidate-1",
            steps=[PlanStep(
                kind="verify",
                description="Verify the issuer terms.",
                evidence_ids=["travel-credit-1"],
                requires_user_confirmation=False,
            )],
            assumptions=[],
        )])

    monkeypatch.setattr(cli, "_run_task", capture_task)
    planner = cli._create_role_callables(None, None, None)["planner"]
    planner("Understand the benefit.", evidence(), [])

    expected = captured["task"].expected_output
    for field_name in ("candidates", "candidate_id", "steps", "kind", "evidence_ids", "requires_user_confirmation", "assumptions"):
        assert field_name in expected
    assert "no more than two" in expected
    assert "{known_evidence_ids}" in captured["task"].description
    assert "fixture:remaining-credit" in captured["inputs"]["known_evidence_ids"]


def test_critic_task_specifies_assessment_json_fields(monkeypatch):
    cli = load_cli()
    captured = {}

    def capture_tasks(_agent, tasks):
        captured["tasks"] = tasks
        return [CriticAssessment(
            candidate_id="candidate-1",
            scores=CandidateScores(grounding=2, goal_fit=2, feasibility=2, clarity=2),
        )]

    monkeypatch.setattr(cli, "_run_critic_tasks", capture_tasks)
    critic = cli._create_role_callables(None, None, None)["critic"]
    critic("Understand the benefit.", evidence(), [CandidatePlan(
        candidate_id="candidate-1",
        steps=[PlanStep(
            kind="verify",
            description="Verify the issuer terms.",
            evidence_ids=["travel-credit-1"],
            requires_user_confirmation=False,
        )],
    )])

    expected = captured["tasks"][0].expected_output
    for field_name in ("candidate_id", "scores", "grounding", "goal_fit", "feasibility", "clarity", "feedback"):
        assert field_name in expected
