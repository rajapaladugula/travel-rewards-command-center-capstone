# Failure modes

Each row is a way this prototype can go wrong, what stops it, and where that is tested. "Residual" is what is not covered. Everything here uses a public benefits snapshot and a synthetic household record only.

| Failure mode | Defense | Tested in | Residual |
|---|---|---|---|
| A related-looking passage answers a question it does not support (for example a "travel" credit answering a points-redemption question) | Distinctive-word gate, then a strict model YES/NO check on every match, strict ones included; unverified means refuse | `test/verified-retrieval.test.mjs`; five-split evaluation (0 false answers for the shipped variant) | A small local model can still say YES wrongly; only 72 synthetic questions were measured |
| The record answers about the wrong benefit ("How much of the Lyft credit is left?") | Questions naming another benefit never use the record route | `test/agent.test.mjs`, `test/server.test.mjs` | Benefit names outside the keyword list would not be recognized |
| The model invents an amount or rule | Amounts come from code (`calculateRemaining`), record answers are built by code, generated text only sees verified passages, and citations are returned with every grounded answer | `test/agent.test.mjs` | Generated prose around cited passages can still paraphrase loosely |
| The app answers confidently on thin evidence | Abstention is the default outcome; the page distinguishes grounded, record, overview, help, and insufficient-evidence states | `test/server.test.mjs`, `test/ui.test.mjs` | Abstains on roughly a third of answerable questions (see README) |
| A question or answer leaves the machine | Ollama URLs must be loopback; judge/rewrite calls validate the same way; no hosted fallback; CrewAI, OpenTelemetry and LangSmith telemetry are disabled at import and a default OpenAI call is pointed at a closed port | `test/local-ollama.test.mjs`, `multi_agent/tests/test_privacy.py` | Disabled by environment switches, not by network enforcement |
| An agent changes a record or takes a financial action | The MCP service exposes exactly three read-only tools; the Planner and Critic have no tools; no write path exists | `test/mcp-server.test.mjs`, `multi_agent/tests/test_models_agents.py` | Record editing is intentionally out of scope |
| A malformed or ungrounded plan is recommended | Deterministic grounding validation, hard-failure gates, a 6/8 score gate, bounded retries, and escalation to human review when no plan passes | `multi_agent/tests/test_validation.py`, `test_graph.py` | Plan quality beyond the gates was measured only on synthetic cases |
| Failures leak prompts or stack traces | Fixed, bounded error messages; unknown exceptions are replaced with a generic one | `test_cli.py`, `test/mcp-server.test.mjs` | Allow-listed messages could still be too specific |
| Results from an earlier question are mistaken for the current one | The page shows a notice when the question text no longer matches the displayed results | `test/ui.test.mjs` | Notice is advisory |
| A request is abused (oversized, malformed, remote) | 4 KiB body limit, content-type and JSON checks, loopback-only binding, path allowlist | `test/server.test.mjs`, `test/ui.test.mjs` | Local single-user prototype, no authentication |

Known limits: the source is one saved public page with five passages; the household is fictional; retrieval and judging use a small local model; the safety evaluation set is small and author-written. None of this is evidence of readiness for real accounts or real financial decisions.
