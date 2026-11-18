# Local Multi-Agent Travel Rewards Capstone

This Python CLI is a local implementation of the Checkpoint 5.1 architecture. It uses three specialist roles: an evidence collector that builds typed facts and citations directly from read-only MCP results, plus CrewAI Planner and Critic roles coordinated by LangGraph. The browser UI remains on the existing single-agent RAG path; this CLI is a separate prototype.

## Requirements

- Node.js 20 or newer and npm
- Python 3.12 and [`uv`](https://docs.astral.sh/uv/)
- Ollama running locally with `llama3.2` and `nomic-embed-text`

## Setup

From the repository root:

```bash
cd experiments/rag-capstone
npm ci
ollama pull llama3.2
ollama pull nomic-embed-text
cd multi_agent
uv sync --python 3.12
```

The CrewAI model endpoint defaults to `http://localhost:11434`. Set `OLLAMA_HOST` only to a loopback HTTP origin such as `http://127.0.0.1:11434`; paths, credentials, and remote hosts are rejected. The CLI does not use a hosted model fallback.

## Run

From the repository root, a direct factual query uses the Evidence and Context Agent only:

```bash
uv run --project experiments/rag-capstone/multi_agent travel-rewards-agent \
  --mode fact \
  --question "How much annual travel credit remains on the synthetic card?"
```

A multi-step request invokes the bounded Planner/Critic workflow:

```bash
uv run --project experiments/rag-capstone/multi_agent travel-rewards-agent \
  --mode plan \
  --question "Given this synthetic record, propose cautious next steps for understanding the remaining travel credit. Identify eligibility or renewal details I should verify; do not assume eligibility or make purchases."
```

The CLI prints a JSON result with cited evidence, candidate plans, validation scores, feedback, and the selected plan. `plan` mode creates at most three candidates per pass and permits at most one critique-driven replan. Deterministic checks reject unlinked assumptions, topic-mismatched citations, unqualified eligibility statements, and currency/percentage values absent from cited evidence; hard failures and Critic feedback go back to the Planner together. If evidence is insufficient or candidates still fail checks, the workflow asks for clarification or returns no recommendation.

## Agent and tool boundaries

- **Evidence and Context Agent:** reads the synthetic card snapshot and retrieves dated public passages; it does not plan or execute actions.
- **Strategy Planner:** proposes typed advisory steps using only the evidence bundle; it has no tools.
- **Critic and Verifier:** checks candidate evidence references and scores grounding, goal fit, feasibility, and clarity; it has no tools.
- **LangGraph controller:** routes `fact` and `plan` modes, validates outputs, resolves ties deterministically, and bounds the feedback loop.
- **MCP stdio server:** exposes only `get_household_snapshot`, `search_benefit_evidence`, and `calculate_remaining_benefit`. These tools are read-only; no card, payment, purchase, or record-update action is available.

## Data and privacy limits

The demo uses only `data/synthetic-household.json` and the public issuer snapshot in `data/chase-sapphire-reserve.md`. That public page is a dated capture from 2026-10-27, not a live statement of current card terms. Do not enter real card, account, household, or other sensitive information. No user-specific memory is persisted. No bank connection, live issuer crawling, or production UI integration is included.

## Synthetic safety evaluation

Run the fixed synthetic pilot from this directory with local Ollama available:

```bash
uv run --python 3.12 python -m travel_rewards_multi_agent.safety_eval
```

The runner uses local Ollama only. It validates 15 labeled synthetic cases (10 development, 5 holdout), compares score gates 6/8, 7/8, and 8/8, and reports preferred-plan agreement, grounded selected-plan rate, unsafe selections, citation support among answered facts and selected plans, escalation accuracy, clarification and fallback rates, plus median/p95 latency separately for each split. Abstentions are excluded from citation support and remain visible in escalation and fallback metrics. It keeps the production gate at 6/8, never tunes it automatically, and emits only case IDs and aggregate/outcome fields—not questions, prompts, or reasoning. A local model/setup failure stops the run; no hosted fallback is used.

The measured 45-call synthetic pilot summary is committed in `evaluation/safety-pilot-results.json`; it contains no scenario prompts or reasoning.

## Tests

```bash
cd experiments/rag-capstone
npm test
cd multi_agent
uv run --python 3.12 pytest -q
```
