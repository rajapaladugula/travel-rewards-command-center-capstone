from __future__ import annotations

import os
from collections.abc import Sequence
from urllib.parse import urlsplit

from crewai import Agent, LLM
from crewai.tools import BaseTool


OLLAMA_MODEL = "ollama/llama3.2"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
APPROVED_MCP_TOOLS = frozenset(
    {
        "get_household_snapshot",
        "search_benefit_evidence",
        "calculate_remaining_benefit",
    }
)


def build_local_llm() -> LLM:
    """Build the configured local Ollama model without a hosted fallback."""
    base_url = (os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
    parsed = urlsplit(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("OLLAMA_HOST must be an HTTP loopback URL to keep model calls local")
    if parsed.port is None:
        raise ValueError("OLLAMA_HOST must include the Ollama port")
    if parsed.path not in {"", "/"} or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("OLLAMA_HOST must be a loopback origin without paths, credentials, or query parameters")

    return LLM(
        model=OLLAMA_MODEL,
        base_url=base_url,
        temperature=0,
        max_tokens=1024,
        context_window_size=4096,
        is_litellm=True,
        additional_params={"num_ctx": 4096},
    )


def build_agents(tools: Sequence[BaseTool], llm: LLM | str) -> tuple[Agent, Agent, Agent]:
    """Create the three specialist agents with least-privilege tool access."""
    tool_list = list(tools)
    names = [getattr(tool, "name", None) for tool in tool_list]
    if len(names) != len(APPROVED_MCP_TOOLS) or set(names) != APPROVED_MCP_TOOLS:
        raise ValueError("Evidence agent accepts only the approved read-only MCP tools")

    evidence_agent = Agent(
        role="Evidence and Context Agent",
        goal=(
            "Retrieve the relevant synthetic household facts and dated public benefit passages, "
            "then return only a concise evidence-grounded answer or a clearly marked evidence gap."
        ),
        backstory=(
            "You are a careful research specialist. Treat tool results as the only source of facts, "
            "attach evidence IDs to every factual answer, and never recommend or execute account actions."
        ),
        llm=llm,
        tools=tool_list,
        allow_delegation=False,
        max_iter=4,
        verbose=False,
    )
    planner_agent = Agent(
        role="Strategy Planner",
        goal=(
            "Propose at most three distinct advisory plans using only the supplied typed evidence; "
            "make assumptions explicit and cite evidence IDs on every non-question step."
        ),
        backstory=(
            "You are a cautious planning specialist. You cannot access tools or alter records; "
            "return only the approved advisory step kinds and request confirmation where needed."
        ),
        llm=llm,
        tools=[],
        allow_delegation=False,
        max_iter=3,
        verbose=False,
    )
    critic_agent = Agent(
        role="Critic and Verifier",
        goal=(
            "Assess every candidate against evidence grounding, the user goal, feasibility, and clarity; "
            "return one structured assessment per candidate with actionable feedback."
        ),
        backstory=(
            "You are an independent verifier. Do not introduce new facts or tools; flag unsupported claims, "
            "missing evidence references, unsafe assumptions, and constraint violations."
        ),
        llm=llm,
        tools=[],
        allow_delegation=False,
        max_iter=3,
        verbose=False,
    )
    return evidence_agent, planner_agent, critic_agent
