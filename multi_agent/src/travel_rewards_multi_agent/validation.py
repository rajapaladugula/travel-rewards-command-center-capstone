from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from .models import CandidatePlan, CandidateReview, EvidenceBundle


ADVISORY_STEP_KINDS = frozenset({"research", "verify", "compare", "calculate", "consider", "ask_user"})
SCORE_DIMENSIONS = ("grounding", "goal_fit", "feasibility", "clarity")
MINIMUM_PLAN_SCORE = 6
_QUANTIFIED_CLAIM = re.compile(
    r"\$\s*((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)|"
    r"\b(\d+(?:\.\d+)?)\s*(?:%|\bpercent\b)(?!\w)",
    re.IGNORECASE,
)
_PERCENT_CLAIM = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:%|\bpercent\b)(?!\w)", re.IGNORECASE)
_ELIGIBILITY_CLAIM = re.compile(r"\b(?:eligib(?:le|ility)|qualif(?:y|ies|ied)|renew(?:al|s|ed|ing)?)\b", re.IGNORECASE)
_QUALIFICATION_CUE = re.compile(r"\b(?:verify|check|confirm|research|ask|inquire|whether|if|potential|possible)\b", re.IGNORECASE)
_TOPIC_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "before", "by", "can", "could", "decide", "decision",
        "details", "for", "from", "in", "information", "into", "is", "it", "its", "may", "might", "of", "on",
        "or", "our", "out", "over", "please", "potential", "public", "research", "review", "step", "terms",
        "that", "the", "their", "this", "to", "up", "use", "using", "user", "verify", "with", "would",
    }
)


def validate_minimum_score(minimum_score: int) -> None:
    if type(minimum_score) is not int or not 0 <= minimum_score <= 8:
        raise ValueError("minimum_score must be an integer from 0 through 8")


def known_evidence_ids(evidence: EvidenceBundle) -> set[str]:
    return {fact.id for fact in evidence.facts} | {citation.id for citation in evidence.citations}


def validate_evidence_bundle(evidence: EvidenceBundle) -> list[str]:
    failures: list[str] = []
    known_ids = known_evidence_ids(evidence)

    if evidence.direct_answer is not None and not evidence.answer_evidence_ids:
        failures.append("direct answer must cite at least one evidence ID")
    if evidence.direct_answer is None and evidence.answer_evidence_ids:
        failures.append("answer evidence IDs cannot be present without a direct answer")
    for evidence_id in evidence.answer_evidence_ids:
        if evidence_id not in known_ids:
            failures.append(f"direct answer references unknown evidence ID: {evidence_id}")

    return failures


def _decimal(value: object) -> Decimal | None:
    try:
        return Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError):
        return None


def _quantity_supported(claim: str, is_percent: bool, evidence_ids: list[str], evidence: EvidenceBundle) -> bool:
    expected = _decimal(claim)
    if expected is None:
        return False

    fact_by_id = {fact.id: fact for fact in evidence.facts}
    citation_by_id = {citation.id: citation for citation in evidence.citations}
    for evidence_id in evidence_ids:
        fact = fact_by_id.get(evidence_id)
        if fact and not is_percent and _decimal(fact.value) == expected:
            return True

        citation = citation_by_id.get(evidence_id)
        if citation is None:
            continue
        citation_text = f"{citation.heading} {citation.excerpt}"
        if is_percent:
            cited_values = _PERCENT_CLAIM.findall(citation_text)
            if any(_decimal(value) == expected for value in cited_values):
                return True
        else:
            cited_values = re.findall(r"\$\s*((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)", citation_text)
            if any(_decimal(value) == expected for value in cited_values):
                return True
    return False


def _topic_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.casefold()) if token not in _TOPIC_STOPWORDS}


def validate_candidate(candidate: CandidatePlan, evidence: EvidenceBundle) -> list[str]:
    failures: list[str] = []
    known_ids = known_evidence_ids(evidence)

    if evidence.household is None:
        failures.append("candidate cannot be checked because household data is missing")
    if candidate.assumptions:
        failures.append(
            "candidate contains assumptions that are not linked to evidence; turn each uncertainty into a verify, research, or ask_user step"
        )

    for index, step in enumerate(candidate.steps, start=1):
        label = f"step {index}"
        referenced_content: list[str] = []
        if step.kind not in ADVISORY_STEP_KINDS:
            failures.append(f"{label} uses a step kind outside the advisory whitelist")
            continue
        if step.kind != "ask_user" and not step.evidence_ids:
            failures.append(f"{label} must cite at least one evidence ID")
        for evidence_id in step.evidence_ids:
            if evidence_id not in known_ids:
                failures.append(f"{label} cites unknown evidence ID: {evidence_id}")
            elif evidence_id in {fact.id for fact in evidence.facts}:
                fact = next(fact for fact in evidence.facts if fact.id == evidence_id)
                referenced_content.append(f"{fact.label} {fact.value}")
            else:
                citation = next(citation for citation in evidence.citations if citation.id == evidence_id)
                referenced_content.append(f"{citation.heading} {citation.excerpt}")
        if len(step.evidence_ids) != len(set(step.evidence_ids)):
            failures.append(f"{label} contains duplicate evidence IDs")
        if step.kind != "ask_user" and referenced_content:
            step_topics = _topic_tokens(step.description)
            if not any(step_topics & _topic_tokens(content) for content in referenced_content):
                failures.append(f"{label} is not related to its cited evidence")

        eligibility_match = _ELIGIBILITY_CLAIM.search(step.description)
        if eligibility_match and not _QUALIFICATION_CUE.search(step.description[:eligibility_match.start()]):
            failures.append(f"{label} states an eligibility or renewal claim instead of framing it as something to verify")

        if step.kind != "ask_user":
            for match in _QUANTIFIED_CLAIM.finditer(step.description):
                claim = match.group(1) or match.group(2)
                is_percent = match.group(2) is not None
                if not _quantity_supported(claim, is_percent, step.evidence_ids, evidence):
                    failures.append(f"{label} makes an unsupported quantitative claim: {match.group(0)}")

    return failures


def score_candidate(
    review: CandidateReview,
    hard_failures: list[str],
    minimum_score: int = MINIMUM_PLAN_SCORE,
) -> int | None:
    validate_minimum_score(minimum_score)
    if hard_failures:
        return None

    values = [getattr(review.scores, name, None) for name in SCORE_DIMENSIONS]
    if any(type(value) is not int or value < 0 or value > 2 for value in values):
        return None

    total = sum(values)
    return total if total >= minimum_score else None
