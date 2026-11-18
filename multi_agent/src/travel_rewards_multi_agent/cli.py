from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from typing import TextIO

from crewai import Crew, Process, Task
from pydantic import ValidationError

from .agents import build_agents, build_local_llm
from .errors import AgentOutputError
from .graph import build_graph
from .mcp_client import default_mcp_server_path, open_mcp_tools
from .models import (
    CandidatePlan,
    Citation,
    CriticAssessment,
    EvidenceBundle,
    EvidenceFact,
    HouseholdSnapshot,
    PlanBatch,
    WorkflowResult,
)
from .validation import MINIMUM_PLAN_SCORE

_SAFE_CLI_ERRORS = {
    "Ollama is unavailable at the local endpoint",
    "OLLAMA_HOST must be an HTTP loopback URL to keep model calls local",
    "OLLAMA_HOST must include the Ollama port",
    "OLLAMA_HOST must be a loopback origin without paths, credentials, or query parameters",
    "A Node.js executable is required to start the MCP server",
}


def format_cli_error(error: Exception) -> str:
    message = str(error)
    if message in _SAFE_CLI_ERRORS:
        return message
    return "Local agent failed safely; check Ollama and MCP setup, then retry."


def _parse_task_output(output, model):
    task_outputs = getattr(output, "tasks_output", None) or []
    task_output = task_outputs[-1] if task_outputs else output
    structured = getattr(task_output, "pydantic", None) or getattr(output, "pydantic", None)
    if isinstance(structured, model):
        return structured
    if structured is not None:
        try:
            return model.model_validate(structured)
        except ValidationError as error:
            raise AgentOutputError(f"CrewAI returned invalid structured output for {model.__name__}") from error

    raw = getattr(task_output, "raw", None)
    if not isinstance(raw, str):
        raise AgentOutputError(f"CrewAI did not return a structured {model.__name__} output")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise AgentOutputError(f"CrewAI did not return valid JSON for {model.__name__}") from error
    if model is PlanBatch:
        if isinstance(payload, list):
            payload = {"candidates": payload}
        if isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
            for candidate in payload["candidates"]:
                if not isinstance(candidate, dict):
                    continue
                candidate.setdefault("assumptions", [])
                if not isinstance(candidate["assumptions"], list):
                    continue
                for step in candidate.get("steps", []):
                    if not isinstance(step, dict):
                        continue
                    for field in ("assumption", "assumptions"):
                        step_assumptions = step.get(field)
                        if isinstance(step_assumptions, str):
                            candidate["assumptions"].append(step.pop(field))
                        elif isinstance(step_assumptions, list) and all(
                            isinstance(item, str) for item in step_assumptions
                        ):
                            candidate["assumptions"].extend(step.pop(field))
    try:
        return model.model_validate(payload)
    except ValidationError as error:
        raise AgentOutputError(f"CrewAI returned invalid structured output for {model.__name__}") from error


def _run_task(agent, task: Task, output_model, inputs: dict[str, object]):
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    output = crew.kickoff(inputs=inputs)
    return _parse_task_output(output, output_model)


def _literal_task_text(value: str) -> str:
    """Escape braces in evidence text so CrewAI's task formatter treats it literally."""
    return value.replace("{", "{{").replace("}", "}}")


def _build_evidence_bundle(question: str, tools) -> EvidenceBundle:
    """Assemble evidence only from read-only MCP results, never model-authored sources."""
    tools_by_name = {tool.name: tool for tool in tools}
    required_tools = {
        "get_household_snapshot",
        "search_benefit_evidence",
        "calculate_remaining_benefit",
    }
    if set(tools_by_name) != required_tools:
        raise RuntimeError("Evidence collection requires the approved read-only MCP tools")

    household_data = json.loads(tools_by_name["get_household_snapshot"].run())
    search_result = json.loads(tools_by_name["search_benefit_evidence"].run(query=question))
    calculation = json.loads(tools_by_name["calculate_remaining_benefit"].run())
    household = {
        **household_data,
        "remainingAmount": calculation.get("remainingAmount"),
    }

    facts = [
        ("fixture:benefit-amount", "benefit amount", calculation["benefitAmount"]),
        ("fixture:used-amount", "used amount", calculation["usedAmount"]),
        ("fixture:remaining-amount", "remaining amount", calculation["remainingAmount"]),
        ("fixture:anniversary-date", "anniversary date", calculation["anniversaryDate"]),
        ("fixture:annual-fee", "annual fee", household_data["annualFee"]),
    ]
    matches = search_result.get("matches")
    if not isinstance(matches, list):
        raise RuntimeError("Evidence search returned an invalid match list")
    citations = [
        Citation(
            id=match["id"],
            heading=match["heading"],
            sourceUrl=match["sourceUrl"],
            capturedAt=match["capturedAt"],
            text=match["text"],
        )
        for match in matches
    ]

    normalized_question = question.casefold()
    asks_remaining = any(term in normalized_question for term in ("remaining", "remain", "left", "balance"))
    asks_benefit = any(term in normalized_question for term in ("benefit", "credit", "travel"))
    if asks_remaining and asks_benefit:
        direct_answer = (
            f"The synthetic record shows ${calculation['remainingAmount']:g} remaining from its "
            f"${calculation['benefitAmount']:g} {calculation['benefitName']} "
            f"(${calculation['usedAmount']:g} used)."
        )
        answer_evidence_ids = ["fixture:remaining-amount"]
    elif citations:
        direct_answer = citations[0].excerpt
        answer_evidence_ids = [citations[0].id]
    else:
        direct_answer = None
        answer_evidence_ids = []

    return EvidenceBundle(
        goal=question,
        household=HouseholdSnapshot.model_validate(household),
        facts=[EvidenceFact(id=fact_id, label=label, value=value) for fact_id, label, value in facts],
        citations=citations,
        missing_data=["Issuer terms come from a dated public snapshot, not live verification."],
        direct_answer=direct_answer,
        answer_evidence_ids=answer_evidence_ids,
    )


def _run_critic_tasks(agent, tasks: list[Task]) -> list[CriticAssessment]:
    if not tasks:
        return []
    crew = Crew(agents=[agent], tasks=tasks, process=Process.sequential, verbose=False)
    output = crew.kickoff()
    task_outputs = getattr(output, "tasks_output", None) or []
    if len(task_outputs) != len(tasks):
        raise RuntimeError("CrewAI returned an incomplete critic task result set")

    assessments = []
    for task_output in task_outputs:
        assessments.append(_parse_task_output(task_output, CriticAssessment))
    return assessments


def _create_role_callables(evidence_agent, planner_agent, critic_agent):
    def evidence_role(question: str) -> EvidenceBundle:
        return _build_evidence_bundle(question, evidence_agent.tools)

    def planner_role(goal: str, evidence: EvidenceBundle, feedback: list[str]) -> PlanBatch:
        known_ids = sorted(
            {fact.id for fact in evidence.facts} | {citation.id for citation in evidence.citations}
        )
        task = Task(
            description=(
                "Create up to three distinct advisory plans for this goal: {goal}\n"
                "Use only the supplied evidence bundle: {evidence_json}\n"
                "Use only these exact evidence IDs in plan steps: {known_evidence_ids}\n"
                "Each step must cite evidence that substantively relates to its topic; include matching key terms from "
                "the cited fact or passage.\n"
                "Apply this critic feedback from the prior pass, if any: {feedback_json}\n"
                "Return a PlanBatch. Each step must use one allowed kind: research, verify, compare, calculate, "
                "consider, or ask_user. Every non-ask_user step must cite known evidence IDs. Do not assert "
                "eligibility or renewal; express unknowns as verify, research, or ask_user steps. Keep assumptions "
                "empty unless each assumption is directly stated in cited evidence. State whether user confirmation "
                "is required. Do not execute actions or invent facts."
            ),
            expected_output=(
                "Return one JSON object, with no wrapper named PlanBatch. It must contain a top-level "
                "candidates array. Each candidate has candidate_id, steps (an array), and assumptions (an array). "
                "Each step has kind, description, evidence_ids (an array of known IDs), and "
                "requires_user_confirmation (a boolean). Put assumptions only in the candidate's assumptions array, never on a step. "
                "Return no more than two concise candidates (never exceed the three-candidate limit); "
                "no markdown or prose."
            ),
            agent=planner_agent,
        )
        return _run_task(
            planner_agent,
            task,
            PlanBatch,
            {
                "goal": goal,
                "evidence_json": evidence.model_dump_json(),
                "feedback_json": json.dumps(feedback, ensure_ascii=False),
                "known_evidence_ids": json.dumps(known_ids, ensure_ascii=False),
            },
        )

    def critic_role(goal: str, evidence: EvidenceBundle, candidates: list[CandidatePlan]) -> list[CriticAssessment]:
        evidence_text = evidence.model_dump_json()
        tasks = []
        for candidate in candidates:
            description = (
                f"Evaluate candidate {candidate.candidate_id} for this goal: {goal}\n"
                f"Evidence bundle: {evidence_text}\n"
                f"Candidate plan: {candidate.model_dump_json()}\n"
                "Return a CriticAssessment for exactly this candidate ID. Score grounding, goal_fit, feasibility, "
                "and clarity from 0 to 2. Add concise actionable feedback. Do not introduce facts or execute actions."
            )
            tasks.append(Task(
                description=_literal_task_text(description),
                expected_output=(
                    "Return one JSON object with candidate_id, scores, and feedback. The scores object must contain "
                    "integer fields grounding, goal_fit, feasibility, and clarity, each from 0 to 2. Feedback is an "
                    "array of concise strings. Do not wrap the object or include markdown or prose."
                ),
                agent=critic_agent,
            ))
        return _run_critic_tasks(critic_agent, tasks)

    return {"evidence": evidence_role, "planner": planner_role, "critic": critic_role}


def run_local_workflow(
    mode: str,
    question: str,
    minimum_score: int = MINIMUM_PLAN_SCORE,
) -> WorkflowResult:
    server_path = default_mcp_server_path()
    with open_mcp_tools("node", server_path) as tools:
        agents = build_agents(tools=tools, llm=build_local_llm())
        roles = _create_role_callables(*agents)
        graph = build_graph(roles, minimum_score=minimum_score)
        state = graph.invoke({"mode": mode, "question": question})
        result = state.get("result")
        if not isinstance(result, WorkflowResult):
            raise RuntimeError("Workflow finished without a validated result")
        return result


def run_cli(
    argv: Sequence[str],
    runtime_factory: Callable[[str, str], WorkflowResult],
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    parser = argparse.ArgumentParser(prog="travel-rewards-agent")
    parser.add_argument("--mode", choices=("fact", "plan"), required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args(list(argv))

    try:
        result = runtime_factory(args.mode, args.question)
        if not isinstance(result, WorkflowResult):
            result = WorkflowResult.model_validate(result)
    except Exception as error:
        stderr.write(f"Travel Rewards agent failed: {format_cli_error(error)}\n")
        return 1

    stdout.write(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n")
    return 1 if result.status == "failed" else 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(
        sys.argv[1:] if argv is None else argv,
        runtime_factory=run_local_workflow,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
