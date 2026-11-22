import test from 'node:test';
import assert from 'node:assert/strict';
import { parseVerdict, sanitizeRewrite, verifiedRetrieve, LOOSE_MIN_SCORE } from '../src/verified-retrieval.mjs';

const card = { product: 'P', issuer: 'I' };
const chunk = (id, score) => ({ id, heading: `H-${id}`, text: `original question text ${id}`, score });

function harness({ strict = [], loose = [], looseAfterRewrite = [], verdicts = {}, rewrite = 'rewritten query' } = {}) {
  const calls = { retrieve: [], judge: [], rewrite: [] };
  const retrieveFn = async ({ query, minScore }) => {
    calls.retrieve.push({ query, minScore });
    if (minScore >= 0.7) return strict;
    return query === rewrite ? looseAfterRewrite : loose;
  };
  const judge = async (question, passage) => { calls.judge.push({ question, id: passage.id }); return verdicts[passage.id] ?? 'no'; };
  const rewriteFn = async (question) => { calls.rewrite.push(question); return rewrite; };
  return { calls, args: { question: 'original question', chunks: [], card, retrieveFn, judge, rewrite: rewriteFn } };
}

test('parses only an explicit leading YES or NO', () => {
  for (const [text, expected] of [['YES', 'yes'], ['Yes.', 'yes'], [' no', 'no'], ['NO, it is only related', 'no'], ['YES, because...', 'yes'], ['Maybe', 'unknown'], ['', 'unknown'], ['NOT sure', 'unknown'], ['yesterday', 'unknown']]) {
    assert.equal(parseVerdict(text), expected, text);
  }
});

test('sanitizes a rewrite to one short line and rejects empty or unchanged output', () => {
  assert.equal(sanitizeRewrite('  "Lyft credit expiration date"\nExtra explanation', 'when does lyft end'), 'Lyft credit expiration date');
  assert.equal(sanitizeRewrite('x'.repeat(500), 'q').length <= 200, true);
  assert.equal(sanitizeRewrite('   ', 'q'), null);
  assert.equal(sanitizeRewrite('Same Question', 'same question'), null);
});

test('with strict verification switched off, a strict hit is returned without judging or rewriting', async () => {
  const { calls, args } = harness({ strict: [chunk('a', 0.8)] });
  const result = await verifiedRetrieve({ ...args, verifyStrict: false });
  assert.equal(result.path, 'strict');
  assert.deepEqual(result.results.map((c) => c.id), ['a']);
  assert.equal(calls.judge.length, 0);
  assert.equal(calls.rewrite.length, 0);
});

test('loose candidates are kept only when the judge says YES', async () => {
  const { calls, args } = harness({ loose: [chunk('a', 0.66), chunk('b', 0.6), chunk('c', 0.55)], verdicts: { b: 'yes' } });
  const result = await verifiedRetrieve(args);
  assert.equal(result.path, 'verified');
  assert.deepEqual(result.results.map((c) => c.id), ['b']);
  assert.equal(calls.rewrite.length, 0);
  assert.ok(calls.retrieve.some((r) => r.minScore === LOOSE_MIN_SCORE));
});

test('rewrites the query at most once and still judges against the ORIGINAL question', async () => {
  const { calls, args } = harness({ loose: [chunk('a', 0.6)], looseAfterRewrite: [chunk('z', 0.62)], verdicts: { z: 'yes' } });
  const result = await verifiedRetrieve(args);
  assert.equal(result.path, 'rewritten');
  assert.deepEqual(result.results.map((c) => c.id), ['z']);
  assert.equal(calls.rewrite.length, 1);
  assert.ok(calls.judge.every((j) => j.question === 'original question'));
});

test('returns nothing when no candidate is verified, even after the rewrite', async () => {
  const { calls, args } = harness({ loose: [chunk('a', 0.6)], looseAfterRewrite: [chunk('z', 0.62)] });
  const result = await verifiedRetrieve(args);
  assert.equal(result.path, 'none');
  assert.deepEqual(result.results, []);
  assert.equal(calls.rewrite.length, 1);
});

test('an unclear judge reply is treated as not verified', async () => {
  const { args } = harness({ loose: [chunk('a', 0.6)], verdicts: { a: 'maybe' } });
  const result = await verifiedRetrieve({ ...args, useRewrite: false });
  assert.deepEqual(result.results, []);
});

test('the rewrite can be switched off and the loose search never goes below its floor', async () => {
  const { calls, args } = harness({ loose: [chunk('a', 0.6)] });
  await verifiedRetrieve({ ...args, useRewrite: false });
  assert.equal(calls.rewrite.length, 0);
  assert.ok(calls.retrieve.every((r) => r.minScore >= LOOSE_MIN_SCORE));
});

test('judge or rewrite failures propagate instead of being treated as safe', async () => {
  const { args } = harness({ loose: [chunk('a', 0.6)] });
  await assert.rejects(() => verifiedRetrieve({ ...args, judge: async () => { throw new Error('Ollama judge request failed'); } }), /Ollama/);
});

import { hasDistinctiveOverlap, contentTokens } from '../src/verified-retrieval.mjs';

const corpus = [
  { id: 'lyft', heading: '$120 in Lyft ride credits', text: 'Get up to $10 in monthly in-app credits to use on rides through September 30, 2027.' },
  { id: 'lounge', heading: 'Lounge access', text: 'Enjoy complimentary access to Priority Pass airport lounges worldwide, with up to two guests.' },
  { id: 'edit', heading: '$500 credit for stays with The Edit', text: 'Receive a statement credit of up to $250 for each prepaid booking with The Edit. A two-night minimum applies. Purchases that qualify will not earn points.' },
  { id: 'travel', heading: '$300 annual travel credit', text: 'Get up to $300 in statement credits for travel purchases each account anniversary year. Purchases that qualify will not earn points.' },
  { id: 'ge', heading: 'Global Entry, TSA PreCheck, or NEXUS fee credit', text: 'Receive one statement credit of up to $120 every four years as reimbursement for the application fee.' },
];
const byId = Object.fromEntries(corpus.map((c) => [c.id, c]));

test('content tokens drop function words and generic card words', () => {
  assert.deepEqual(contentTokens('Can I add authorized users to this card?').sort(), ['add', 'authorized', 'user']);
});

test('a candidate passes only when it shares a distinctive word with the question', () => {
  assert.equal(hasDistinctiveOverlap('Is there a minimum stay for The Edit?', byId.edit, corpus), true);
  assert.equal(hasDistinctiveOverlap('What is the benefit of the lounge?', byId.lounge, corpus), true);
  assert.equal(hasDistinctiveOverlap('When does the Lyft credit expire?', byId.lyft, corpus), true);
  assert.equal(hasDistinctiveOverlap('Can I add authorized users?', byId.edit, corpus), false);
  assert.equal(hasDistinctiveOverlap('Does the card cover baggage delay?', byId.lounge, corpus), false);
});

test('a word that appears in most passages is not distinctive on its own', () => {
  assert.equal(hasDistinctiveOverlap('What credit do I get?', byId.travel, corpus), false);
});

test('candidates without distinctive overlap are dropped before the judge is asked', async () => {
  const judged = [];
  const result = await verifiedRetrieve({
    question: 'Can I add authorized users?',
    chunks: corpus,
    card,
    retrieveFn: async ({ minScore }) => (minScore >= 0.7 ? [] : [{ ...byId.edit, score: 0.6 }]),
    judge: async (_question, passage) => { judged.push(passage.id); return 'YES'; },
    rewrite: async () => null,
  });
  assert.deepEqual(result.results, []);
  assert.deepEqual(judged, []);
});

test('by default a strict hit is also verified, and an unverified strict hit is refused', async () => {
  const accepted = harness({ strict: [chunk('a', 0.8)], verdicts: { a: 'yes' } });
  const ok = await verifiedRetrieve(accepted.args);
  assert.equal(ok.path, 'strict');
  assert.deepEqual(ok.results.map((c) => c.id), ['a']);
  assert.equal(accepted.calls.judge.length, 1);

  const rejected = harness({ strict: [chunk('a', 0.75)] });
  const none = await verifiedRetrieve({ ...rejected.args, useRewrite: false });
  assert.equal(none.path, 'none');
  assert.deepEqual(none.results, []);
});

test('a rejected strict hit can still be recovered by a verified loose match', async () => {
  const { args } = harness({ strict: [chunk('a', 0.75)], loose: [chunk('a', 0.75), chunk('b', 0.6)], verdicts: { b: 'yes' } });
  const result = await verifiedRetrieve(args);
  assert.equal(result.path, 'verified');
  assert.deepEqual(result.results.map((c) => c.id), ['b']);
});

import { parseSelection, selectVerified } from '../src/verified-retrieval.mjs';

test('parses a selection as a candidate number, or none', () => {
  assert.equal(parseSelection('2', 3), 2);
  assert.equal(parseSelection('Passage 1.', 3), 1);
  assert.equal(parseSelection('NONE', 3), null);
  assert.equal(parseSelection('none of them', 3), null);
  assert.equal(parseSelection('4', 3), null);
  assert.equal(parseSelection('0', 3), null);
  assert.equal(parseSelection('', 3), null);
  assert.equal(parseSelection('1 and 2', 3), null);
});

test('selection mode offers the strict and loose candidates together, gated by distinctive overlap, and keeps only the chosen one', async () => {
  const offered = [];
  const result = await selectVerified({
    question: 'Is there a minimum stay for The Edit?',
    chunks: corpus,
    card,
    retrieveFn: async ({ minScore }) => (minScore >= 0.7 ? [{ ...byId.travel, score: 0.71 }] : [{ ...byId.travel, score: 0.71 }, { ...byId.edit, score: 0.66 }, { ...byId.lounge, score: 0.6 }]),
    select: async (_question, candidates) => { offered.push(candidates.map((c) => c.id)); return String(candidates.findIndex((c) => c.id === 'edit') + 1); },
    rewrite: async () => null,
  });
  assert.deepEqual(result.results.map((c) => c.id), ['edit']);
  assert.deepEqual(offered, [['edit']]); // travel and lounge share no distinctive word with the question
});

test('selection mode returns nothing for NONE, and skips the model when no candidate passes the overlap gate', async () => {
  let asked = 0;
  const base = { chunks: corpus, card, retrieveFn: async () => [{ ...byId.edit, score: 0.6 }], rewrite: async () => null, useRewrite: false };
  const none = await selectVerified({ ...base, question: 'Is there a minimum stay for The Edit?', select: async () => { asked += 1; return 'NONE'; } });
  assert.deepEqual(none.results, []);
  assert.equal(asked, 1);
  const gated = await selectVerified({ ...base, question: 'Can I add authorized users?', select: async () => { asked += 1; return '1'; } });
  assert.deepEqual(gated.results, []);
  assert.equal(asked, 1);
});
