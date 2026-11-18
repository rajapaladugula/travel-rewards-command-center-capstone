# Checkpoint 5.1: Multi-Agent Architecture and Coordination Plan

## Purpose and rationale

The Travel Rewards Command Center is a learning prototype for household travel-rewards planning. Its intended user wants to understand a synthetic card-benefit record, compare it with public issuer terms, and get cautious next steps. The agent must distinguish household data from policy, cite sources, expose gaps, and avoid unsupported suggestions. It uses synthetic data and a dated public snapshot, with no financial connections or actions.

A single prompt could retrieve facts, plan, and approve its own plan. These tasks need different standards: evidence gathering needs provenance, planning explores options, and review challenges assumptions. Separate roles make handoffs visible and let deterministic checks reject unsupported steps. The coordination cost is justified for planning requests; simple factual questions use only the evidence path.

## Specialist team

Three roles provide enough separation for evidence, planning, and review without a larger team’s overhead.

- **Evidence and Context Agent** reads the synthetic household snapshot, retrieves relevant public passages, and obtains deterministic benefit arithmetic. It assembles a typed evidence bundle with stable fact/citation IDs, source URL and capture date, missing-data flags, and a concise answer only when supported. It does not plan or take financial actions.
- **Strategy Planner** receives the user’s goal and evidence, then proposes up to three advisory plans. Steps use an allowed type (`research`, `verify`, `compare`, `calculate`, `consider`, or `ask_user`) and cite evidence unless asking the user. Unverified eligibility or renewal details become questions or verification steps.
- **Critic and Verifier** scores each candidate for grounding, goal and constraint fit, feasibility, and clarity, and provides concise feedback. Deterministic controller checks—not the Critic’s prose or score alone—enforce hard constraints.

## Workflow and communication

A deterministic LangGraph controller runs a hybrid workflow: evidence gathering and review are sequential, with one bounded feedback loop. Factual requests route through Evidence only. Planning requests pass the evidence bundle to Planner, then up to three candidates to Critic. The controller rejects unknown or topic-mismatched citations, unsupported monetary/percentage claims, unlinked assumptions, unqualified eligibility claims, and unsafe step types before scoring. Eligible candidates receive 0–2 points on four dimensions and need at least 6/8. If none qualify, the controller sends deterministic failures plus Critic feedback to Planner for one replan. If candidates still fail, it asks for clarification or returns no recommendation; ties go to the first candidate.

Agents exchange typed EvidenceBundle, PlanBatch, and CriticAssessment structures, not unrestricted private reasoning transcripts. Stable IDs connect proposed steps to facts or citations. The evidence collector assembles read-only MCP results; CrewAI runs the model-based Planner and Critic; LangGraph, in the LangChain ecosystem, manages state and feedback.

## Environment and trade-offs

The environment includes the user’s question, synthetic card fixture, dated public issuer snapshot, local Ollama models, and local Node MCP server. MCP exposes only `get_household_snapshot`, `search_benefit_evidence`, and `calculate_remaining_benefit`; calculations are deterministic. No real account data, durable personal memory, or mutation tools are used. Users see citations, proposed steps, reviews, and evidence gaps, then decide what to do outside the prototype.

Role separation and validation improve reliability. Candidate and retry limits control latency, complexity, and coordination overhead; the three-role design scales to more planning requests without adding a specialist for every subtask. Costs include extra model calls and malformed outputs, so typed validation fails closed. The browser UI remains on the earlier single-agent RAG path; this local CLI is a separate prototype, not an integrated UI workflow. If a future version edits saved records, it will preview old and proposed values side by side and require explicit confirmation; this prototype has no write path.