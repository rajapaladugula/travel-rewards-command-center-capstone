"""Privacy switches for the agent frameworks, applied when the package is imported (before CrewAI loads).

- CrewAI anonymous telemetry and CrewAI cloud tracing: off.
- OpenTelemetry (any exporter): off.
- LangSmith / LangChain tracing: off, and no endpoint or API key is kept in the environment.
- OpenAI: nothing in this project uses it. A stray default-LLM call is pointed at a closed local port (the
  discard port), so it fails loudly instead of reaching a cloud API.

The project's only model endpoint is the loopback Ollama service; these switches keep the frameworks from adding
any other network channel.
"""
from __future__ import annotations

import os

FORCED = {
    "CREWAI_DISABLE_TELEMETRY": "true",
    "CREWAI_TRACING_ENABLED": "false",
    "OTEL_SDK_DISABLED": "true",
    "OTEL_TRACES_EXPORTER": "none",
    "OTEL_METRICS_EXPORTER": "none",
    "OTEL_LOGS_EXPORTER": "none",
    "LANGCHAIN_TRACING_V2": "false",
    "LANGCHAIN_TRACING": "false",
    "LANGSMITH_TRACING": "false",
    "OPENAI_BASE_URL": "http://127.0.0.1:9/openai-disabled",
    "OPENAI_API_BASE": "http://127.0.0.1:9/openai-disabled",
}
REMOVED = (
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGSMITH_ENDPOINT",
    "LANGCHAIN_ENDPOINT",
    "OPENAI_API_KEY",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
)


def configure() -> None:
    os.environ.update(FORCED)
    for key in REMOVED:
        os.environ.pop(key, None)


def problems() -> list[str]:
    """Settings that are not as required; an empty list means the lockdown is intact."""
    found = [f"{key} should be {value!r}" for key, value in FORCED.items() if os.environ.get(key) != value]
    found += [f"{key} must not be set" for key in REMOVED if os.environ.get(key)]
    return found
