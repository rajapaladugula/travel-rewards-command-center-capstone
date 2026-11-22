import json
import os
import subprocess
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")


def run_fresh(code: str, env: dict[str, str] | None = None) -> dict:
    """Run code in a fresh interpreter so import-time behavior is what is tested."""
    environment = {k: v for k, v in os.environ.items() if not k.startswith(("CREWAI_", "OTEL_", "LANGCHAIN_", "LANGSMITH_", "OPENAI_"))}
    environment.update(env or {})
    environment["PYTHONPATH"] = SRC
    completed = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, check=True, timeout=120)
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_importing_the_package_disables_framework_telemetry_before_crewai_loads():
    result = run_fresh(
        "import json, os\n"
        "import travel_rewards_multi_agent\n"
        "import crewai\n"
        "print(json.dumps({k: os.environ.get(k) for k in ['CREWAI_DISABLE_TELEMETRY','CREWAI_TRACING_ENABLED','OTEL_SDK_DISABLED','OTEL_TRACES_EXPORTER','LANGCHAIN_TRACING_V2','LANGSMITH_TRACING']}))"
    )
    assert result == {
        "CREWAI_DISABLE_TELEMETRY": "true",
        "CREWAI_TRACING_ENABLED": "false",
        "OTEL_SDK_DISABLED": "true",
        "OTEL_TRACES_EXPORTER": "none",
        "LANGCHAIN_TRACING_V2": "false",
        "LANGSMITH_TRACING": "false",
    }


def test_user_environment_cannot_turn_telemetry_back_on_or_leak_cloud_keys():
    result = run_fresh(
        "import json, os\n"
        "import travel_rewards_multi_agent\n"
        "print(json.dumps({k: os.environ.get(k) for k in ['CREWAI_DISABLE_TELEMETRY','LANGSMITH_API_KEY','LANGCHAIN_API_KEY','OPENAI_API_KEY','OTEL_EXPORTER_OTLP_ENDPOINT']}))",
        env={
            "CREWAI_DISABLE_TELEMETRY": "false",
            "LANGSMITH_API_KEY": "secret-1",
            "LANGCHAIN_API_KEY": "secret-2",
            "OPENAI_API_KEY": "secret-3",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "https://collector.example",
        },
    )
    assert result == {
        "CREWAI_DISABLE_TELEMETRY": "true",
        "LANGSMITH_API_KEY": None,
        "LANGCHAIN_API_KEY": None,
        "OPENAI_API_KEY": None,
        "OTEL_EXPORTER_OTLP_ENDPOINT": None,
    }


def test_default_openai_endpoint_points_at_a_closed_local_port_so_stray_calls_fail():
    result = run_fresh(
        "import json, os\n"
        "import travel_rewards_multi_agent\n"
        "print(json.dumps({'base': os.environ.get('OPENAI_BASE_URL'), 'api': os.environ.get('OPENAI_API_BASE')}))"
    )
    assert result["base"].startswith("http://127.0.0.1:9/")
    assert result["api"].startswith("http://127.0.0.1:9/")


def test_problems_reports_nothing_after_configuration_and_flags_drift():
    from travel_rewards_multi_agent import privacy

    privacy.configure()
    assert privacy.problems() == []
    os.environ["CREWAI_DISABLE_TELEMETRY"] = "false"
    try:
        assert any("CREWAI_DISABLE_TELEMETRY" in problem for problem in privacy.problems())
    finally:
        privacy.configure()
