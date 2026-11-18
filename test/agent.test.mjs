import test from 'node:test';
import assert from 'node:assert/strict';
import { answerQuestion, calculateRemaining } from '../src/agent.mjs';

const household = {
  card: {
    product: 'Chase Sapphire Reserve',
    issuer: 'Chase',
    benefitName: 'annual travel credit',
    benefitAmount: 300,
    usedAmount: 180,
    anniversaryDate: '08-15',
  },
};
const source = {
  id: 'travel-credit-1',
  product: 'Chase Sapphire Reserve',
  issuer: 'Chase',
  heading: '$300 annual travel credit',
  text: 'Get up to $300 in statement credits for travel purchases each account anniversary year.',
  sourceUrl: 'https://creditcards.chase.com/rewards-credit-cards/sapphire/reserve',
  capturedAt: '2026-10-27',
};

test('calculates remaining benefit value deterministically', () => {
  assert.equal(calculateRemaining({ benefitAmount: 300, usedAmount: 180 }), 120);
  assert.equal(calculateRemaining({ benefitAmount: 300, usedAmount: 300 }), 0);
});

test('rejects invalid or overused benefit amounts', () => {
  assert.throws(() => calculateRemaining({ benefitAmount: -1, usedAmount: 0 }), /non-negative numbers/i);
  assert.throws(() => calculateRemaining({ benefitAmount: 300, usedAmount: 301 }), /cannot exceed the benefit amount/i);
});

test('no-retrieval baseline reports the amount but does not invent a reset rule', async () => {
  const result = await answerQuestion({
    question: 'When does my travel credit reset?',
    household,
    retrievedChunks: [],
    retrieveEnabled: false,
    generate: async () => assert.fail('baseline must not call the language model'),
  });

  assert.match(result.answer, /\$120 remaining/i);
  assert.match(result.answer, /cannot determine.*reset rule/i);
  assert.deepEqual(result.citations, []);
});

test('retrieval-grounded response supplies retrieved context and cites its source', async () => {
  let generationInput;
  const result = await answerQuestion({
    question: 'Is the annual travel credit based on a calendar year or an account-anniversary year?',
    household,
    retrievedChunks: [source],
    retrieveEnabled: true,
    generate: async (input) => {
      generationInput = input;
      return 'The credit applies each account-anniversary year.';
    },
  });

  assert.match(generationInput, /account anniversary year/i);
  assert.match(generationInput, /calendar year or an account-anniversary year/i);
  assert.match(generationInput, /if evidence says each account anniversary year, answer that cycle directly/i);
  assert.match(generationInput, /do not infer a specific date/i);
  assert.match(generationInput, /120/);
  assert.match(result.answer, /account-anniversary year/i);
  assert.equal(result.citations[0].sourceUrl, source.sourceUrl);
  assert.equal(result.citations[0].capturedAt, source.capturedAt);
  assert.equal(result.citations[0].heading, source.heading);
});

test('abstains without calling the generator when retrieval has no supporting passage', async () => {
  const result = await answerQuestion({
    question: 'Is there a gas-station category bonus?',
    household,
    retrievedChunks: [],
    retrieveEnabled: true,
    generate: async () => assert.fail('unsupported questions must abstain before generation'),
  });

  assert.match(result.answer, /insufficient evidence/i);
  assert.deepEqual(result.citations, []);
});
