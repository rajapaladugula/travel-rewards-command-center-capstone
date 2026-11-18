import json
import re
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1]
SUBMISSION = PACKAGE_ROOT / "safety-checkpoint-6-1.md"
README = PACKAGE_ROOT / "README.md"
CHANGE = Path(__file__).parents[4] / "specs/001-travel-rewards-command-center/changes/CHG-004-safety-evaluation.md"
SPEC = Path(__file__).parents[4] / "specs/001-travel-rewards-command-center/capstone-rag-demo/multi-agent/spec.md"
PILOT_REPORT = PACKAGE_ROOT / "evaluation" / "safety-pilot-results.json"


def words(text):
    return re.findall(r"\b[\w$./:-]+\b", text)


def test_checkpoint_matches_word_count_and_covers_safety_requirements_and_measured_results():
    text = SUBMISSION.read_text()
    normalized = text.casefold()
    assert 400 <= len(words(text)) <= 600
    for phrase in (
        "15-case", "10 development", "5 holdout", "preferred-plan match rate",
        "grounded selected-plan rate", "unsafe selected-plan count", "escalation accuracy",
        "clarification rate", "median latency", "p95 latency", "fallback success rate",
        "measured pilot results", "0 unsafe selected plans", "loopback", "no hosted fallback",
        "human review", "synthetic", "side by side", "read-only",
    ):
        assert phrase in normalized, f"checkpoint is missing: {phrase}"
    assert "future version" in normalized
    assert "not statistical calibration" in normalized


def test_readme_documents_the_local_evaluation_command_without_hosted_fallback():
    text = README.read_text().casefold()
    assert "python -m travel_rewards_multi_agent.safety_eval" in text
    assert "local ollama only" in text
    assert "10 development" in text and "5 holdout" in text


def test_spec_and_change_record_capture_the_implemented_evaluation_contract():
    spec = SPEC.read_text().casefold()
    change = CHANGE.read_text().casefold()
    for phrase in ("6/8", "7/8", "8/8", "15 synthetic", "10 development", "5 holdout"):
        assert phrase in spec
        assert phrase in change
    for phrase in ("npm test", "pytest", "unsafe selected plans", "development", "holdout", "latency"):
        assert phrase in change


def test_committed_pilot_report_is_prompt_free_and_matches_checkpoint_results():
    report = json.loads(PILOT_REPORT.read_text())
    assert report["production_threshold"] == 6
    assert len(report["case_results"]) == 45
    assert all("question" not in row for row in report["case_results"])
    assert all("citation_supported" in row for row in report["case_results"])
    for threshold in ("6", "7", "8"):
        assert report["thresholds"][threshold]["development"]["preferred_plan_match_rate"] == 0
        assert report["thresholds"][threshold]["holdout"]["preferred_plan_match_rate"] == 0
        assert report["thresholds"][threshold]["development"]["unsafe_selected_plan_count"] == 0
        assert report["thresholds"][threshold]["holdout"]["unsafe_selected_plan_count"] == 0
        assert report["thresholds"][threshold]["development"]["grounded_selected_plan_rate"] is None
        assert report["thresholds"][threshold]["holdout"]["citation_support_rate"] is None
