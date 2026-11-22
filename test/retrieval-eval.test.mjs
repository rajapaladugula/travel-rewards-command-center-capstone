import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { loadCorpus } from '../src/corpus.mjs';
import { evaluateRetrieval } from '../src/retrieval-eval.mjs';

const casesFile = new URL('../evaluation/retrieval-cases.json', import.meta.url);

test('labeled retrieval cases are well formed and match real source headings', async () => {
  const { cases } = JSON.parse(await readFile(casesFile, 'utf8'));
  const headings = new Set((await loadCorpus()).map((chunk) => chunk.heading));
  assert.equal(cases.length, 72);
  assert.equal(new Set(cases.map((c) => c.id)).size, cases.length);
  const split = (name, labeled) => cases.filter((c) => c.split === name && Boolean(c.expectedHeading) === labeled).length;
  assert.deepEqual([split('development', true), split('development', false), split('holdout', true), split('holdout', false), split('holdout2', true), split('holdout2', false), split('holdout3', true), split('holdout3', false), split('holdout4', true), split('holdout4', false)], [17, 7, 6, 6, 6, 6, 6, 6, 6, 6]);
  assert.ok(cases.every((c) => ['development', 'holdout', 'holdout2', 'holdout3', 'holdout4'].includes(c.split)));
  for (const c of cases) {
    assert.ok(c.question.trim().length > 0 && c.question.length <= 300, c.id);
    if (c.expectedHeading) assert.ok(headings.has(c.expectedHeading), `${c.id}: unknown heading`);
  }
});

test('metrics separate correct, wrong, and abstained answers for answerable and unsupported questions', async () => {
  const cases = [
    { id: 'a', question: 'q1', expectedHeading: 'H1' },
    { id: 'b', question: 'q2', expectedHeading: 'H1' },
    { id: 'c', question: 'q3', expectedHeading: 'H2' },
    { id: 'd', question: 'q4', expectedHeading: null },
    { id: 'e', question: 'q5', expectedHeading: null },
  ];
  const fake = { q1: [{ heading: 'H1' }], q2: [], q3: [{ heading: 'H1' }], q4: [], q5: [{ heading: 'H2' }] };
  const result = await evaluateRetrieval({ cases, chunks: [], card: { product: 'P', issuer: 'I' }, minScore: 0.7, retrieveFn: async ({ query }) => fake[query] });
  assert.deepEqual(result.answerable, { total: 3, correct: 1, wrongPassage: 1, abstained: 1 });
  assert.deepEqual(result.unsupported, { total: 2, abstained: 1, falseAnswers: 1 });
  assert.equal(result.answerableRecall, 1 / 3);
  assert.deepEqual(result.cases.map((c) => [c.id, c.outcome]), [['a', 'correct'], ['b', 'abstained'], ['c', 'wrong-passage'], ['d', 'correct-abstention'], ['e', 'false-answer']]);
  assert.equal(JSON.stringify(result).includes('q1'), false, 'report must not contain question text');
});

import { compareVariants } from '../src/retrieval-eval.mjs';

test('compares strict, verified, and rewrite variants separately on each split without leaking question text', async () => {
  const cases = [
    { id: 'd1', question: 'dev answerable', expectedHeading: 'H1', split: 'development' },
    { id: 'd2', question: 'dev unsupported', expectedHeading: null, split: 'development' },
    { id: 'h1', question: 'hold answerable', expectedHeading: 'H1', split: 'holdout' },
    { id: 'h2', question: 'hold unsupported', expectedHeading: null, split: 'holdout' },
  ];
  const chunks = [{ id: 'c1', heading: 'H1', text: 'answerable text', product: 'P', issuer: 'I', score: 0.6 }];
  const retrieveFn = async ({ minScore }) => (minScore >= 0.7 ? [] : chunks);
  const judge = async (question) => (question.includes('answerable') ? 'YES' : 'NO');
  const rewrite = async (question) => `${question} rewritten`;

  const report = await compareVariants({ cases, chunks, card: { product: 'P', issuer: 'I' }, retrieveFn, judge, rewrite });

  assert.deepEqual(Object.keys(report.variants), ['strict', 'verified-loose', 'verified-all']);
  assert.equal(report.variants.strict.development.answerable.correct, 0);
  assert.equal(report.variants['verified-loose'].development.answerable.correct, 1);
  assert.equal(report.variants['verified-loose'].holdout.answerable.correct, 1);
  assert.equal(report.variants['verified-loose'].holdout.unsupported.falseAnswers, 0);
  assert.equal(report.variants['verified-all'].development.unsupported.abstained, 1);
  assert.equal(JSON.stringify(report).includes('answerable'), true); // metric names only
  assert.equal(JSON.stringify(report).includes('dev answerable'), false);
});
