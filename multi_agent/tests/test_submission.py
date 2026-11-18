from pathlib import Path
import re

import pytest


SUBMISSION_PATH = Path(__file__).resolve().parents[1] / "submission.md"


def test_checkpoint_submission_has_required_scope_and_word_count():
    if not SUBMISSION_PATH.exists():
        pytest.fail("verified Checkpoint 5.1 submission has not been written", pytrace=False)

    text = SUBMISSION_PATH.read_text(encoding="utf-8")
    normalized = text.lower()
    words = re.findall(r"\b[\w’'-]+\b", text)

    assert 400 <= len(words) <= 600
    for required_phrase in (
        "intended user",
        "single-agent",
        "evidence and context agent",
        "strategy planner",
        "critic and verifier",
        "coordination",
        "typed",
        "crewai",
        "langgraph",
        "mcp",
        "reliability",
        "complexity",
        "coordination overhead",
        "public",
        "synthetic",
        "browser ui",
    ):
        assert required_phrase in normalized, f"missing required submission coverage: {required_phrase}"
