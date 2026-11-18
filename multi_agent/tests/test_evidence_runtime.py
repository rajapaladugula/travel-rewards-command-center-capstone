import json
from types import SimpleNamespace

from travel_rewards_multi_agent.cli import _build_evidence_bundle
from travel_rewards_multi_agent.models import EvidenceBundle


HOUSEHOLD = {
    "product": "Chase Sapphire Reserve",
    "issuer": "Chase",
    "annualFee": 795,
    "benefitName": "annual travel credit",
    "benefitAmount": 300,
    "usedAmount": 180,
    "anniversaryDate": "08-15",
}
CALCULATION = {
    "benefitName": "annual travel credit",
    "benefitAmount": 300,
    "usedAmount": 180,
    "remainingAmount": 120,
    "anniversaryDate": "08-15",
    "currency": "USD",
}
PASSAGE = {
    "id": "travel-credit-public-1",
    "heading": "$300 annual travel credit",
    "text": "Get up to $300 in statement credits for travel purchases each account anniversary year.",
    "score": 0.81,
    "sourceUrl": "https://creditcards.chase.com/rewards-credit-cards/sapphire/reserve",
    "capturedAt": "2026-10-27",
}


def fake_tools(matches=None):
    calls = []

    def tool(name, result):
        def run(**kwargs):
            calls.append((name, kwargs))
            return json.dumps(result)

        return SimpleNamespace(name=name, run=run)

    tools = [
        tool("get_household_snapshot", HOUSEHOLD),
        tool("search_benefit_evidence", {"matches": matches or []}),
        tool("calculate_remaining_benefit", CALCULATION),
    ]
    return tools, calls


def test_evidence_bundle_uses_real_mcp_values_and_retrieved_citations_only():
    tools, calls = fake_tools([PASSAGE])

    result = _build_evidence_bundle("How much annual travel credit remains?", tools)

    assert isinstance(result, EvidenceBundle)
    assert result.household.remaining_amount == 120
    assert result.direct_answer == (
        "The synthetic record shows $120 remaining from its $300 annual travel credit ($180 used)."
    )
    assert result.answer_evidence_ids == ["fixture:remaining-amount"]
    assert [citation.id for citation in result.citations] == ["travel-credit-public-1"]
    assert result.citations[0].source_url == PASSAGE["sourceUrl"]
    assert [name for name, _kwargs in calls] == [
        "get_household_snapshot",
        "search_benefit_evidence",
        "calculate_remaining_benefit",
    ]
    assert calls[1][1] == {"query": "How much annual travel credit remains?"}


def test_evidence_bundle_abstains_on_empty_retrieval_for_non_fixture_questions():
    tools, _calls = fake_tools()

    result = _build_evidence_bundle("What gasoline benefit does this card provide?", tools)

    assert result.direct_answer is None
    assert result.answer_evidence_ids == []
    assert result.citations == []


def test_evidence_bundle_answers_policy_question_with_exact_retrieved_passage():
    tools, _calls = fake_tools([PASSAGE])

    result = _build_evidence_bundle("When does the travel credit apply?", tools)

    assert result.direct_answer == PASSAGE["text"]
    assert result.answer_evidence_ids == [PASSAGE["id"]]
    assert result.citations[0].excerpt == PASSAGE["text"]
