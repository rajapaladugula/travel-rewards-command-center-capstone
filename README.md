# Travel Rewards RAG Capstone Demo

A local, read-only demonstration of retrieval-augmented answers for the Travel Rewards Personal Assistant. Use the CLI or browser UI to compare the same question with and without retrieval. The sample combines a public issuer-benefits snapshot with a synthetic card record. It does not connect to accounts, modify records, or make financial decisions.

## Requirements

- Node.js 20 or later
- Ollama running locally
- `llama3.2` for answer generation and `nomic-embed-text` for local embeddings

If needed, install the models once:

```sh
ollama pull llama3.2
ollama pull nomic-embed-text
```

The public source snapshot is saved at `data/chase-sapphire-reserve.md`; it is not refreshed automatically. Check its capture date and the cited issuer page before using the example as current benefit advice. All household values in `data/synthetic-household.json` are fictional.

## Install dependencies

From the repository root, install the pinned Node.js dependencies once after checkout and again whenever the lockfile changes:

```sh
npm ci
```

## Run the CLI

```sh
npm test
npm run demo
```

The CLI asks the same cycle question first without retrieval and then with retrieval. It shows the passages that pass the relevance filter, citations, and a deterministic remaining-credit calculation. A separate unrelated query demonstrates abstention. The vector ranking is in memory because this example contains only a handful of chunks; a hosted vector database is unnecessary for the prototype.

## Run the browser UI

```sh
npm run ui
```

Open [http://127.0.0.1:4173](http://127.0.0.1:4173). Use the sample questions or enter a question and select **Compare answers**. The page displays the baseline and grounded response, the deterministic calculation, and the retrieved passages with score and capture metadata. When no public passage matches, the grounded answer can still respond in two bounded ways: facts that live in the synthetic household record (credit remaining, amount used, annual fee, account-anniversary date) are answered by code and labeled as coming from the record, and questions about what the assistant can do get a plain description of its scope. Broad overview questions such as "What are the benefits of this card?" list every section of the saved source (built by code, with citations) because no single passage matches them. Anything else, such as a gasoline bonus or whether the credit covers hotels, still abstains. Step 03, **Multi-agent review**, runs the same question through the Python multi-agent workflow: choose *Fact lookup* (Evidence agent only) or *Cautious plan* (Evidence → Planner → Critic, which can take up to a minute) and select **Run multi-agent workflow**. The page shows the status, answer, any candidate plans with their 0–2 Critic scores and hard-failure reasons, and the cited sources. When the safety gates reject every candidate it says no plan was recommended and that human review is required. This step needs the Python environment from `multi_agent/` (`uv sync --python 3.12`). Press `Ctrl+C` in the terminal to stop the server. The UI server binds only to `127.0.0.1`; it does not persist questions or change records.

The embedding and generation requests stay on the local Ollama service. `OLLAMA_HOST`, when set, must be an HTTP loopback origin (`localhost`, `127.0.0.0/8`, or `::1`) without credentials, paths, queries, or fragments; remote endpoints are rejected before sending data. If Ollama or a model is unavailable, the CLI or UI reports an error; neither falls back to a hosted API. Unit tests use fake vectors/responses and do not require Ollama.

## Verified retrieval and its evaluation

Ordinary questions go through four checks before any passage reaches the answer: (1) similarity search at the 0.70 cutoff, plus a looser 0.50 search for rephrasings; (2) a deterministic gate that keeps a passage only if it shares a word that is distinctive in the source (at most two of the five passages contain it), so a model cannot approve a passage with nothing in common with the question; (3) a strict one-word YES/NO check by the local `llama3.2` model, run for strict matches too; and (4) at most one rewrite of the query, always checked against the user's original question. Anything unverified is dropped and the app refuses. Help, record, and overview questions are answered by code and never reach retrieval; a question that names a different benefit (for example "How much of the Lyft credit is left?") is never answered from the travel-credit record.

`npm run eval:retrieval` compares three variants (plus one rejected candidate) on 72 synthetic questions in five splits (`evaluation/retrieval-cases.json`): `development` (24, used while building), and four holdouts of 12 written and frozen *before* the design step they test. Each holdout that exposed a failure became a regression set, and the success rule was fixed in advance: ship a variant only with zero false answers and zero wrong passages on every split. Pooled over all five splits (41 answerable, 31 unsupported):

| Variant | Answerable answered correctly | Wrong passages | Unsupported refused | False answers |
|---|---|---|---|---|
| `strict` (0.70 cutoff only) | 23 / 41 | 0 | 27 / 31 | 4 |
| `verified-all` (**shipped**) | 26 / 41 | 0 | 31 / 31 | 0 |
| `select-all` (rejected) | 31 / 41 | 0 | 30 / 31 | 1 |

The honest reading: verification mainly buys **safety** (four unsupported questions that the cutoff alone answered from an unrelated passage are now refused) and only a small coverage gain; the app still abstains on about a third of answerable questions. `select-all`, where the model picks between candidates, answers more but gave one false answer ("What travel purchases earn extra points?" answered from a related passage), so it was not shipped. The model checks are non-deterministic in principle (temperature 0, but local-model dependent), the questions are short synthetic ones written by one author, and the source has only five passages, so these are design evidence, not statistical calibration. The report stores case IDs and counts only, not question text. Run `npm run eval:retrieval -- --splits development` to evaluate one split without overwriting the committed full report.

## Multi-agent workflow (CLI and browser)

This package also contains a Python multi-agent workflow in [`multi_agent/`](multi_agent/). The browser UI calls it through `POST /api/agents`, which runs the same `travel-rewards-agent` CLI as a child process (no shell, one run at a time, 240-second timeout) and returns only the validated result fields; the CLI and the page therefore share one workflow and one set of safety gates. CrewAI defines Evidence, Planner, and Critic roles; LangGraph controls the bounded workflow; and the Node MCP server supplies only read-only snapshot, public-evidence-search, and deterministic-calculation tools. The browser's comparison panel (steps 01–02) remains the single-agent RAG path, and step 03 adds the multi-agent workflow to the same page.

Follow [`multi_agent/README.md`](multi_agent/README.md) to install Python 3.12/`uv`, run the CLI, and execute the fixed 15-case safety evaluation. The latest pilot compared 6/8, 7/8, and 8/8 score thresholds; it selected no plans and is not evidence of deployment readiness. The prompt-free measured report is available at [`multi_agent/evaluation/safety-pilot-results.json`](multi_agent/evaluation/safety-pilot-results.json); see the nested README for metric definitions and limitations. All test fixtures and evaluation cases are synthetic; no personal or account data is included.
