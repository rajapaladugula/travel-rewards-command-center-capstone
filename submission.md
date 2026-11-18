# Capstone Checkpoint 3.1: RAG and Retrieval Design

## Retrieval decision and rationale

Retrieval is required for the Travel Rewards Personal Assistant because part of the answer depends on issuer benefit terms that are not reliably known to the model and may change over time. The household-specific facts—such as which card is held, how much of a credit has been used, and a member’s account-anniversary date—are different: they are exact records, so the agent should retrieve them from structured application data rather than use semantic search. RAG is appropriate for external, unstructured policy information; it is not a replacement for the product’s records or calculation rules.

For this checkpoint I built a small, local command-line demonstration, not a production feature or the full assistant. It uses only a publicly available issuer page and synthetic household values. It does not access a bank account, update records, or make a financial decision.

## Retrieval source and design

The external source is Chase’s public [Sapphire Reserve benefits page](https://creditcards.chase.com/rewards-credit-cards/sapphire/reserve). I saved the relevant sections as a dated snapshot, captured 2026-10-27, so the demonstration is reproducible and the user can see that the issuer’s terms may have changed since capture. The snapshot includes benefit headings and source metadata, not information from a private account.

The page is segmented at benefit headings so terms and conditions stay together. Longer sections are split at sentence boundaries into chunks of about 250–450 estimated tokens with a small overlap. Each chunk carries product, issuer, heading, source URL, and capture date. At query time, the local `nomic-embed-text` model creates query and passage embeddings through Ollama. The demo ranks passages by cosine similarity, filters for the exact product and issuer, considers at most the top three, and applies a configurable minimum score initially set to 0.70. It returned an in-memory vector index rather than adding a hosted vector database: the small, fixed corpus does not need a separate database service. The local `llama3.2` model composes an answer from the retrieved context; source citations and arithmetic are supplied by application code.

## Observable effect of retrieval

The demonstration asks: “Is the annual travel credit based on a calendar year or an account-anniversary year?” Its synthetic card record contains a $300 annual travel credit, $180 already used, and an August 15 anniversary date.

Without retrieval, the assistant can calculate $120 remaining from the record, but says it cannot determine the benefit cycle from those values alone. With retrieval, the query returns the public-source chunk headed “$300 annual travel credit,” with a similarity score of 0.795. That passage states that the credit applies “each account anniversary year.” The answer then identifies the cycle as account-anniversary based, reports the separately calculated $120 remaining, and shows the issuer URL and capture date. The other candidate chunks scored below the 0.70 threshold and were not passed to the model. An unrelated question about gasoline bonuses retrieves no passage above the threshold, so the agent returns an insufficient-evidence response rather than inventing a benefit.

## Failure mode and mitigation

One retrieval failure is using a stale page or returning a similar but irrelevant benefit passage. The snapshot date makes freshness visible but does not guarantee the issuer’s terms are still current. Exact product/issuer filtering, a relevance threshold, and showing the matched passage and source help the user detect a mismatch. When no passage reaches the threshold, the system abstains instead of asking the language model to answer from general knowledge. Numeric values remain in structured synthetic data and are calculated deterministically, reducing the risk that model prose changes the arithmetic. The local model can still misinterpret relevant wording, so this small demonstration does not establish production reliability; its answer is paired with inspectable source evidence, and questions should not ask it to infer a specific date the source does not state.

## Improvement over prompt-only design

A prompt-only model cannot retrieve the issuer’s current benefit wording unless that text is supplied. In this example it cannot distinguish calendar-year and account-anniversary language from the synthetic balance record alone. Retrieval changes the response by adding a verifiable issuer passage to the agent’s context, while the structured record and deterministic calculation provide the household-specific amount. This division of work addresses a concrete grounding failure without treating retrieved text as a source of private account data or allowing the assistant to take consequential actions.