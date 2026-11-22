import test from 'node:test';
import assert from 'node:assert/strict';
import { answerQuestion, calculateRemaining, isOverviewQuestion } from '../src/agent.mjs';

const household = {
  card: {
    product: 'Chase Sapphire Reserve',
    issuer: 'Chase',
    benefitName: 'annual travel credit',
    benefitAmount: 300,
    usedAmount: 180,
    annualFee: 795,
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

test('answers remaining-amount questions from the synthetic record when no public passage is needed', async () => {
  for (const question of ['how much credit is left?', 'How much of the travel credit remains?', 'What is my remaining balance?']) {
    const result = await answerQuestion({
      question,
      household,
      retrievedChunks: [],
      retrieveEnabled: true,
      generate: async () => assert.fail('record-based amounts are calculated by code, not generated'),
    });
    assert.equal(result.source, 'household-record', question);
    assert.match(result.answer, /\$120 remaining/);
    assert.match(result.answer, /\$300/);
    assert.match(result.answer, /\$180 used/);
    assert.match(result.answer, /synthetic record/i);
    assert.deepEqual(result.citations, []);
  }
});

test('still abstains on questions the record cannot answer, even with amount-like wording', async () => {
  for (const question of ['Does this card earn a bonus on gasoline purchases?', 'Is there a gas-station category bonus?', 'When does the credit reset?']) {
    const result = await answerQuestion({
      question,
      household,
      retrievedChunks: [],
      retrieveEnabled: true,
      generate: async () => assert.fail('unsupported questions must abstain before generation'),
    });
    assert.notEqual(result.source, 'household-record', question);
    assert.match(result.answer, /insufficient evidence/i, question);
  }
});

const noGenerate = () => assert.fail('record and help answers must not call the generator');
const ask = (question, extra = {}) => answerQuestion({ question, household, retrievedChunks: [], retrieveEnabled: true, generate: noGenerate, ...extra });

test('explains what the assistant can and cannot do for capability questions', async () => {
  for (const question of ['what can you help me with?', 'What do you do?', 'How can you help me']) {
    const result = await ask(question);
    assert.equal(result.source, 'app-help', question);
    assert.match(result.answer, /benefit/i);
    assert.match(result.answer, /synthetic/i);
    assert.match(result.answer, /cannot|does not|doesn't/i);
    assert.deepEqual(result.citations, []);
  }
});

test('answers annual fee, anniversary date, and amount used from the synthetic record', async () => {
  const fee = await ask("What's my annual fee?");
  assert.equal(fee.source, 'household-record');
  assert.match(fee.answer, /\$795/);

  const anniversary = await ask('When is my account anniversary date?');
  assert.equal(anniversary.source, 'household-record');
  assert.match(anniversary.answer, /August 15/);

  const used = await ask('How much of the travel credit have I used?');
  assert.equal(used.source, 'household-record');
  assert.match(used.answer, /\$180/);
  for (const result of [fee, anniversary, used]) assert.match(result.answer, /synthetic (household )?record/i);
});

test('does not invent a record fact that the record lacks', async () => {
  const noFee = { card: { ...household.card, annualFee: undefined } };
  const result = await ask("What's my annual fee?", { household: noFee });
  assert.notEqual(result.source, 'household-record');
  assert.match(result.answer, /insufficient evidence/i);
});

test('record and help routing does not capture unsupported questions or cycle questions', async () => {
  for (const question of ['Does this card earn a bonus on gasoline purchases?', 'When does the credit reset?', 'Is there a lounge in Paris?']) {
    const result = await ask(question);
    assert.notEqual(result.source, 'household-record', question);
    assert.notEqual(result.source, 'app-help', question);
    assert.match(result.answer, /insufficient evidence/i, question);
  }
});

test('retrieved evidence still takes priority over record answers', async () => {
  const result = await answerQuestion({
    question: 'Is the credit based on the account anniversary year?',
    household,
    retrievedChunks: [source],
    retrieveEnabled: true,
    generate: async () => 'It applies each account-anniversary year.',
  });
  assert.equal(result.source, undefined);
  assert.equal(result.citations.length, 1);
});

test('words like "used" or "balance" in other kinds of questions do not trigger a record answer', async () => {
  for (const question of ['Can the travel credit be used for hotels?', 'What is the balance transfer fee?', 'Is the lounge used by guests?', 'Can I get a left-handed card?']) {
    const result = await ask(question);
    assert.notEqual(result.source, 'household-record', question);
    assert.match(result.answer, /insufficient evidence/i, question);
  }
  for (const question of ['How much have I used?', 'How much did I spend of the credit?', 'what is left of my credit', 'How much credit remains']) {
    assert.equal((await ask(question)).source, 'household-record', question);
  }
});

const sections = [
  { id: 'a', product: 'Chase Sapphire Reserve', issuer: 'Chase', heading: '$300 annual travel credit', text: 'Get up to $300 in statement credits.', sourceUrl: source.sourceUrl, capturedAt: '2026-10-27' },
  { id: 'b', product: 'Chase Sapphire Reserve', issuer: 'Chase', heading: 'Lounge access', text: 'Complimentary lounge access.', sourceUrl: source.sourceUrl, capturedAt: '2026-10-27' },
];

test('lists the saved benefit sections for overview questions without calling the generator', async () => {
  const result = await answerQuestion({ question: 'what are the benefits of this card?', household, retrievedChunks: sections, retrieveEnabled: true, overview: true, generate: noGenerate });
  assert.equal(result.source, 'benefit-overview');
  assert.match(result.answer, /2 benefits/);
  assert.match(result.answer, /\$300 annual travel credit/);
  assert.match(result.answer, /Lounge access/);
  assert.match(result.answer, /2026-10-27/);
  assert.deepEqual(result.citations.map((c) => c.heading), ['$300 annual travel credit', 'Lounge access']);
});

test('recognizes overview questions narrowly', () => {
  for (const q of ['what are the benefits of this card?', 'What benefits does this card include?', 'List the perks and credits', 'what are all the benefits']) assert.equal(isOverviewQuestion(q), true, q);
  for (const q of ['Does this card earn a bonus on gasoline purchases?', 'Can the travel credit be used for hotels?', 'How much credit is left?', 'What is the benefit of the lounge?', 'Is the lounge a benefit I can use in Paris?']) assert.equal(isOverviewQuestion(q), false, q);
});

import { classifyQuestion } from '../src/agent.mjs';

test('classifies questions that need no retrieval, and leaves everything else to verified retrieval', () => {
  assert.equal(classifyQuestion('what can you help me with?', household.card), 'help');
  assert.equal(classifyQuestion('what are the benefits of this card?', household.card), 'overview');
  assert.equal(classifyQuestion('How much credit is left?', household.card), 'record');
  assert.equal(classifyQuestion("What's my annual fee?", household.card), 'record');
  assert.equal(classifyQuestion('Is the credit based on the account anniversary year?', household.card), null);
  assert.equal(classifyQuestion('Does this card earn a bonus on gasoline purchases?', household.card), null);
});

test('the record route never answers about a different benefit', () => {
  for (const question of ['How much of the Lyft credit is left?', 'What is the remaining Edit credit?', 'How much have I used of my lounge access?', 'How much of the Global Entry credit remains?']) {
    assert.equal(classifyQuestion(question, household.card), null, question);
  }
});
