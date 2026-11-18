import { loadCorpus, loadSyntheticHousehold } from './corpus.mjs';
import { retrieve } from './retriever.mjs';
import { answerQuestion } from './agent.mjs';

const household = await loadSyntheticHousehold();
const chunks = await loadCorpus();
const card = household.card;
const question = 'Is the annual travel credit based on a calendar year or an account-anniversary year?';

console.log('TRAVEL REWARDS RAG CAPSTONE DEMO');
console.log('Synthetic data only; no account changes are made.\n');
console.log(`Question: ${question}\n`);

const baseline = await answerQuestion({ question, household, retrieveEnabled: false });
console.log('--- Without retrieval ---');
console.log(baseline.answer);

const retrieved = await retrieve({
  query: question,
  chunks,
  product: card.product,
  issuer: card.issuer,
});
console.log('\n--- Retrieved evidence ---');
if (!retrieved.length) {
  console.log('No passage met the relevance threshold.');
} else {
  for (const [index, chunk] of retrieved.entries()) {
    console.log(`[S${index + 1}] ${chunk.heading} (score ${chunk.score.toFixed(3)})`);
    console.log(chunk.text);
    console.log(`Source: ${chunk.sourceUrl} (captured ${chunk.capturedAt})\n`);
  }
}

const grounded = await answerQuestion({
  question,
  household,
  retrievedChunks: retrieved,
  retrieveEnabled: true,
});
console.log('--- With retrieval ---');
console.log(grounded.answer);
console.log(`Deterministic calculation: $${card.benefitAmount} limit − $${card.usedAmount} used = $${grounded.remainingAmount} remaining.`);
console.log('Sources:');
for (const citation of grounded.citations) {
  console.log(`- ${citation.heading}: ${citation.sourceUrl} (captured ${citation.capturedAt})`);
}

const unrelatedQuestion = 'Does this card earn a bonus on gasoline purchases?';
const unrelated = await retrieve({
  query: unrelatedQuestion,
  chunks,
  product: card.product,
  issuer: card.issuer,
});
const abstention = await answerQuestion({
  question: unrelatedQuestion,
  household,
  retrievedChunks: unrelated,
  retrieveEnabled: true,
});
console.log('\n--- Unrelated question / retrieval guard ---');
console.log(abstention.answer);
