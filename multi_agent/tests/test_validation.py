from importlib import import_module

import pytest

from travel_rewards_multi_agent.models import (
    CandidatePlan,
    CandidateReview,
    CandidateScores,
    Citation,
    EvidenceBundle,
    EvidenceFact,
    HouseholdSnapshot,
    PlanStep,
)


def load_validation():
    try:
        return import_module("travel_rewards_multi_agent.validation")
    except ModuleNotFoundError:
        pytest.fail("deterministic validation is not implemented", pytrace=False)


def make_evidence(missing_data=None):
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
        citations=[Citation(
            id="travel-credit-1",
            heading="Travel Credit",
            source_url="https://creditcards.chase.com/rewards-credit-cards/sapphire/reserve",
            captured_at="2026-10-27",
            excerpt="The travel credit applies during the account anniversary year.",
        )],
        missing_data=missing_data or [],
    )


def make_candidate(*, candidate_id="candidate-1", evidence_ids=None, kind="verify"):
    return CandidatePlan(
        candidate_id=candidate_id,
        steps=[PlanStep(
            kind=kind,
            description="Verify the travel-credit terms using the cited account-anniversary evidence.",
            evidence_ids=evidence_ids if evidence_ids is not None else ["travel-credit-1"],
            requires_user_confirmation=True,
        )],
        assumptions=[],
    )


def test_evidence_bundle_validation_accepts_only_known_answer_references():
    validation = load_validation()
    evidence = make_evidence().model_copy(update={
        "direct_answer": "The synthetic record shows $120 remaining.",
        "answer_evidence_ids": ["fixture:remaining-credit"],
    })

    assert validation.validate_evidence_bundle(evidence) == []
    invalid = evidence.model_copy(update={"answer_evidence_ids": ["unknown-source"]})
    assert "unknown evidence ID" in " ".join(validation.validate_evidence_bundle(invalid))


def test_candidate_validation_rejects_unknown_evidence_ids():
    validation = load_validation()

    failures = validation.validate_candidate(
        make_candidate(evidence_ids=["unknown-source"]),
        make_evidence(),
    )

    assert any("unknown evidence ID" in failure for failure in failures)


def test_candidate_validation_rejects_non_question_steps_without_references():
    validation = load_validation()
    candidate = make_candidate().model_copy(update={
        "steps": [make_candidate().steps[0].model_copy(update={"evidence_ids": []})]
    })

    failures = validation.validate_candidate(candidate, make_evidence())

    assert any("must cite" in failure for failure in failures)


def test_candidate_validation_allows_unreferenced_user_question_steps():
    validation = load_validation()
    candidate = CandidatePlan(
        candidate_id="question-1",
        steps=[PlanStep(
            kind="ask_user",
            description="What travel dates should the plan consider?",
            evidence_ids=[],
            requires_user_confirmation=True,
        )],
        assumptions=[],
    )

    assert validation.validate_candidate(candidate, make_evidence()) == []


def test_candidate_fails_when_the_structured_household_snapshot_is_missing():
    validation = load_validation()
    evidence = make_evidence().model_copy(update={"household": None})

    failures = validation.validate_candidate(
        make_candidate(kind="calculate"),
        evidence,
    )

    assert any("household data is missing" in failure for failure in failures)


def test_candidate_with_unverified_assumptions_is_not_eligible_for_selection():
    validation = load_validation()
    candidate = make_candidate().model_copy(update={
        "assumptions": ["The current issuer terms are up-to-date."],
    })

    failures = validation.validate_candidate(candidate, make_evidence())

    assert any("assumptions" in failure for failure in failures)


@pytest.mark.parametrize(
    "description,should_fail",
    [
        ("The card is eligible for renewal.", True),
        ("Verify whether the card is eligible for renewal.", False),
    ],
)
def test_candidate_must_frame_eligibility_as_an_open_question(description, should_fail):
    validation = load_validation()
    step = make_candidate().steps[0].model_copy(update={"description": description})
    candidate = make_candidate().model_copy(update={"steps": [step]})

    failures = validation.validate_candidate(candidate, make_evidence())

    assert any("eligibility" in failure for failure in failures) is should_fail


@pytest.mark.parametrize(
    "description,should_fail",
    [
        ("Consider spending $500 on travel.", True),
        ("Consider using the remaining $120 credit.", False),
        ("This benefit is worth 60% of the annual fee.", True),
    ],
)
def test_candidate_cannot_assert_unreferenced_monetary_or_percentage_values(description, should_fail):
    validation = load_validation()
    step = make_candidate().steps[0].model_copy(update={"description": description})
    if "$120" in description:
        step = step.model_copy(update={"evidence_ids": ["fixture:remaining-credit"]})
    candidate = make_candidate().model_copy(update={"steps": [step]})

    failures = validation.validate_candidate(candidate, make_evidence())

    assert any("quantitative claim" in failure for failure in failures) is should_fail


@pytest.mark.parametrize(
    "description,evidence_ids,should_fail",
    [
        ("Research issuer renewal details.", ["fixture:remaining-credit"], True),
        ("Verify the travel credit terms for the account anniversary year.", ["travel-credit-1"], False),
    ],
)
def test_candidate_step_must_share_a_topic_with_at_least_one_cited_evidence_item(
    description, evidence_ids, should_fail
):
    validation = load_validation()
    step = make_candidate().steps[0].model_copy(update={
        "description": description,
        "evidence_ids": evidence_ids,
    })
    candidate = make_candidate().model_copy(update={"steps": [step]})

    failures = validation.validate_candidate(candidate, make_evidence())

    assert any("not related to its cited evidence" in failure for failure in failures) is should_fail


def test_unrelated_missing_data_does_not_block_a_supported_deterministic_calculation():
    validation = load_validation()
    evidence = make_evidence(missing_data=["planned travel date"])
    supported_calculation = make_candidate(kind="calculate", evidence_ids=["fixture:remaining-credit"])
    supported_step = supported_calculation.steps[0].model_copy(update={
        "description": "Calculate the remaining credit using the cited synthetic amount.",
    })
    supported_calculation = supported_calculation.model_copy(update={"steps": [supported_step]})

    assert validation.validate_candidate(supported_calculation, evidence) == []


def test_candidate_score_requires_six_points_and_no_hard_failures():
    validation = load_validation()
    score_six = CandidateReview(
        candidate_id="candidate-1",
        scores=CandidateScores(grounding=2, goal_fit=2, feasibility=1, clarity=1),
    )
    score_five = CandidateReview(
        candidate_id="candidate-2",
        scores=CandidateScores(grounding=1, goal_fit=1, feasibility=1, clarity=2),
    )
    rejected = CandidateReview(
        candidate_id="candidate-3",
        scores=CandidateScores(grounding=2, goal_fit=2, feasibility=2, clarity=2),
        hard_failures=["unknown evidence ID"],
    )

    assert validation.score_candidate(score_six, []) == 6
    assert validation.score_candidate(score_five, []) is None
    assert validation.score_candidate(rejected, rejected.hard_failures) is None


@pytest.mark.parametrize(
    "score,minimum_score,expected",
    [(score, minimum, score if score >= minimum else None)
     for score in (6, 7, 8)
     for minimum in (6, 7, 8)],
)
def test_candidate_score_applies_configured_threshold(score, minimum_score, expected):
    validation = load_validation()
    review = CandidateReview(
        candidate_id="candidate-1",
        scores=CandidateScores(
            grounding=2,
            goal_fit=2,
            feasibility=2,
            clarity=score - 6,
        ),
    )

    assert validation.score_candidate(review, [], minimum_score=minimum_score) == expected


@pytest.mark.parametrize("minimum_score", [-1, 9, 6.0, True, "6"])
def test_candidate_score_rejects_invalid_threshold(minimum_score):
    validation = load_validation()
    review = CandidateReview(
        candidate_id="candidate-1",
        scores=CandidateScores(grounding=2, goal_fit=2, feasibility=1, clarity=1),
    )

    with pytest.raises(ValueError, match="minimum_score"):
        validation.score_candidate(review, [], minimum_score=minimum_score)


def test_candidate_score_hard_failure_overrides_every_valid_threshold():
    validation = load_validation()
    review = CandidateReview(
        candidate_id="candidate-1",
        scores=CandidateScores(grounding=2, goal_fit=2, feasibility=2, clarity=2),
    )

    assert validation.score_candidate(review, ["unsupported claim"], minimum_score=0) is None
