from importlib import import_module

import pytest


def load_models():
    try:
        return import_module("travel_rewards_multi_agent.models")
    except ModuleNotFoundError:
        pytest.fail("typed agent contracts are not implemented", pytrace=False)


def validation_error():
    from pydantic import ValidationError

    return ValidationError


def household_data():
    return {
        "product": "Chase Sapphire Reserve",
        "issuer": "Chase",
        "annual_fee": 795,
        "benefit_name": "annual travel credit",
        "benefit_amount": 300,
        "used_amount": 180,
        "anniversary_date": "08-15",
        "remaining_amount": 120,
    }


def valid_plan(models, candidate_id="candidate-1"):
    return models.CandidatePlan(
        candidate_id=candidate_id,
        steps=[models.PlanStep(
            kind="verify",
            description="Verify eligible travel purchases against the issuer terms.",
            evidence_ids=["travel-credit-1"],
            requires_user_confirmation=False,
        )],
        assumptions=[],
    )


def test_valid_household_and_evidence_bundle_parse_with_grounded_direct_answer():
    models = load_models()
    household = models.HouseholdSnapshot(**household_data())
    bundle = models.EvidenceBundle(
        goal="Understand the remaining synthetic travel credit.",
        household=household,
        facts=[models.EvidenceFact(id="fixture:remaining-credit", label="remaining credit", value=120)],
        citations=[models.Citation(
            id="travel-credit-1",
            heading="Travel Credit",
            source_url="https://creditcards.chase.com/rewards-credit-cards/sapphire/reserve",
            captured_at="2026-10-27",
            excerpt="The travel credit applies during the account anniversary year.",
        )],
        missing_data=[],
        direct_answer="The synthetic record shows $120 remaining.",
        answer_evidence_ids=["fixture:remaining-credit"],
    )

    assert household.remaining_amount == 120
    assert bundle.direct_answer == "The synthetic record shows $120 remaining."
    assert bundle.answer_evidence_ids == ["fixture:remaining-credit"]


def test_evidence_contract_accepts_camel_case_mcp_fields_and_passage_text():
    models = load_models()
    household = models.HouseholdSnapshot.model_validate({
        "product": "Chase Sapphire Reserve",
        "issuer": "Chase",
        "annualFee": 795,
        "benefitName": "annual travel credit",
        "benefitAmount": 300,
        "usedAmount": 180,
        "anniversaryDate": "08-15",
        "remainingAmount": 120,
    })
    citation = models.Citation.model_validate({
        "id": "travel-credit-1",
        "heading": "Travel Credit",
        "sourceUrl": "https://creditcards.chase.com/rewards-credit-cards/sapphire/reserve",
        "capturedAt": "2026-10-27",
        "text": "The travel credit applies during the account anniversary year.",
    })

    assert household.annual_fee == 795
    assert household.remaining_amount == 120
    assert citation.source_url.startswith("https://")
    assert citation.captured_at == "2026-10-27"
    assert citation.excerpt.startswith("The travel credit")


def test_household_rejects_negative_or_inconsistent_amounts_and_invalid_anniversary():
    models = load_models()

    with pytest.raises(validation_error()):
        models.HouseholdSnapshot(**{**household_data(), "benefit_amount": -1})
    with pytest.raises(validation_error()):
        models.HouseholdSnapshot(**{**household_data(), "used_amount": 301})
    with pytest.raises(validation_error()):
        models.HouseholdSnapshot(**{**household_data(), "anniversary_date": "2026-08-15"})


def test_candidate_scores_reject_values_outside_zero_to_two():
    models = load_models()

    scores = models.CandidateScores(grounding=0, goal_fit=2, feasibility=1, clarity=2)
    assert scores.grounding == 0
    assert scores.clarity == 2
    with pytest.raises(validation_error()):
        models.CandidateScores(grounding=3, goal_fit=1, feasibility=1, clarity=1)


def test_non_user_advisory_steps_require_an_allowed_kind_and_evidence_ids():
    models = load_models()

    step = models.PlanStep(
        kind="verify",
        description="Verify the eligibility terms.",
        evidence_ids=["travel-credit-1"],
        requires_user_confirmation=False,
    )
    assert step.kind == "verify"
    with pytest.raises(validation_error()):
        models.PlanStep(kind="purchase", description="Buy a ticket", evidence_ids=["travel-credit-1"], requires_user_confirmation=True)
    with pytest.raises(validation_error()):
        models.PlanStep(kind="verify", description="Verify eligibility", evidence_ids=[], requires_user_confirmation=False)
    assert models.PlanStep(kind="ask_user", description="Which trip?", evidence_ids=[], requires_user_confirmation=True).kind == "ask_user"


def test_evidence_bundle_rejects_unknown_direct_answer_evidence_ids():
    models = load_models()

    with pytest.raises(validation_error()):
        models.EvidenceBundle(
            goal="Understand the remaining synthetic travel credit.",
            household=models.HouseholdSnapshot(**household_data()),
            facts=[models.EvidenceFact(id="fixture:remaining-credit", label="remaining credit", value=120)],
            citations=[],
            missing_data=[],
            direct_answer="There is $120 remaining.",
            answer_evidence_ids=["unknown-source"],
        )


def test_planner_batch_rejects_more_than_three_candidates():
    models = load_models()
    candidates = [valid_plan(models, f"candidate-{index}") for index in range(4)]

    with pytest.raises(validation_error()):
        models.PlanBatch(candidates=candidates)


def test_workflow_result_retains_candidate_steps_for_user_review():
    models = load_models()
    evidence = models.EvidenceBundle(
        goal="Plan around the synthetic travel credit.",
        household=models.HouseholdSnapshot(**household_data()),
        facts=[models.EvidenceFact(id="fixture:remaining-credit", label="remaining credit", value=120)],
        citations=[models.Citation(
            id="travel-credit-1",
            heading="Travel Credit",
            source_url="https://creditcards.chase.com/rewards-credit-cards/sapphire/reserve",
            captured_at="2026-10-27",
            excerpt="The travel credit applies during the account anniversary year.",
        )],
        missing_data=[],
    )
    plan = valid_plan(models)
    review = models.CandidateReview(
        candidate_id="candidate-1",
        scores=models.CandidateScores(grounding=2, goal_fit=2, feasibility=1, clarity=1),
    )

    result = models.WorkflowResult(
        mode="plan",
        status="answered",
        answer="Review the eligible travel terms before deciding.",
        evidence=evidence,
        candidates=[plan],
        candidate_reviews=[review],
        selected_candidate_id="candidate-1",
        retry_count=0,
    )

    assert result.candidates[0].steps[0].description == plan.steps[0].description


def test_local_llm_uses_configured_ollama_host_and_never_selects_a_hosted_model(monkeypatch):
    load_models()
    agents_module = import_module("travel_rewards_multi_agent.agents")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

    llm = agents_module.build_local_llm()

    assert llm.model == "ollama/llama3.2"
    assert llm.provider == "ollama"
    assert llm.is_litellm is True
    assert llm.base_url == "http://127.0.0.1:11434"
    assert llm.context_window_size == 4096
    assert llm.max_tokens == 1024
    assert llm.additional_params["num_ctx"] == 4096


@pytest.mark.parametrize(
    "host",
    [
        "http://localhost:11434/v1",
        "http://user:secret@localhost:11434",
        "http://ollama.example.com:11434",
    ],
)
def test_local_llm_rejects_non_origin_credentialed_or_remote_hosts(monkeypatch, host):
    load_models()
    agents_module = import_module("travel_rewards_multi_agent.agents")
    monkeypatch.setenv("OLLAMA_HOST", host)

    with pytest.raises(ValueError):
        agents_module.build_local_llm()


def test_agent_factory_creates_three_roles_with_tools_only_for_evidence_agent():
    models = load_models()
    from crewai.tools import BaseTool
    from pydantic import Field

    agents_module = import_module("travel_rewards_multi_agent.agents")

    class StubTool(BaseTool):
        name: str
        description: str
        marker: str = Field(default="test")

        def _run(self, **kwargs):
            return self.marker

    tool_names = [
        "get_household_snapshot",
        "search_benefit_evidence",
        "calculate_remaining_benefit",
    ]
    tools = [StubTool(name=name, description=f"Read-only {name} test tool.") for name in tool_names]
    agents = agents_module.build_agents(tools=tools, llm="test/local-stub")

    assert len(agents) == 3
    evidence, planner, critic = agents
    assert "Evidence" in evidence.role
    assert "Planner" in planner.role
    assert "Critic" in critic.role
    assert {tool.name for tool in evidence.tools} == set(tool_names)
    assert planner.tools == []
    assert critic.tools == []

    with pytest.raises(ValueError, match="only the approved read-only MCP tools"):
        agents_module.build_agents(
            tools=[StubTool(name="charge_card", description="Unapproved mutation tool")],
            llm="test/local-stub",
        )
