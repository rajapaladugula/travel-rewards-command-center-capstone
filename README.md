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

Open [http://127.0.0.1:4173](http://127.0.0.1:4173). Use the sample questions or enter a question and select **Compare answers**. The page displays the baseline and grounded response, the deterministic calculation, and the retrieved passages with score and capture metadata. Press `Ctrl+C` in the terminal to stop the server. The UI server binds only to `127.0.0.1`; it does not persist questions or change records.

The embedding and generation requests stay on the local Ollama service. `OLLAMA_HOST`, when set, must be an HTTP loopback origin (`localhost`, `127.0.0.0/8`, or `::1`) without credentials, paths, queries, or fragments; remote endpoints are rejected before sending data. If Ollama or a model is unavailable, the CLI or UI reports an error; neither falls back to a hosted API. Unit tests use fake vectors/responses and do not require Ollama.

## Multi-agent capstone CLI

This package also contains a separate Python multi-agent workflow in [`multi_agent/`](multi_agent/). CrewAI defines Evidence, Planner, and Critic roles; LangGraph controls the bounded workflow; and the Node MCP server supplies only read-only snapshot, public-evidence-search, and deterministic-calculation tools. The browser UI above remains the independent single-agent RAG path.

Follow [`multi_agent/README.md`](multi_agent/README.md) to install Python 3.12/`uv`, run the CLI, and execute the fixed 15-case safety evaluation. The latest pilot compared 6/8, 7/8, and 8/8 score thresholds; it selected no plans and is not evidence of deployment readiness. The prompt-free measured report is available at [`multi_agent/evaluation/safety-pilot-results.json`](multi_agent/evaluation/safety-pilot-results.json); see the nested README for metric definitions and limitations. All test fixtures and evaluation cases are synthetic; no personal or account data is included.
