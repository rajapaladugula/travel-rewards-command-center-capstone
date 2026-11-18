import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer, runQuestion, startServer } from '../src/server.mjs';

const comparison = {
  baseline: 'Cannot determine the benefit cycle without retrieval.',
  grounded: 'The credit applies each account-anniversary year.',
  remainingAmount: 120,
  evidence: [{ id: 'travel-credit', heading: '$300 annual travel credit', text: 'Each account anniversary year.', score: 0.795 }],
  citations: [{ heading: '$300 annual travel credit', sourceUrl: 'https://creditcards.chase.com/rewards-credit-cards/sapphire/reserve', capturedAt: '2026-10-27' }],
};

async function usingServer(server, callback) {
  await new Promise((resolve, reject) => server.once('error', reject).listen(0, '127.0.0.1', resolve));
  try {
    const { port } = server.address();
    return await callback(`http://127.0.0.1:${port}`);
  } finally {
    await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  }
}

test('POST /api/ask returns the baseline, grounded answer, calculation, and evidence', async () => {
  let receivedQuestion;
  const server = createServer({ runQuestion: async (question) => { receivedQuestion = question; return comparison; } });

  await usingServer(server, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/ask`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ question: 'Is the credit calendar-year or anniversary-year?' }),
    });
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), comparison);
    assert.equal(receivedQuestion, 'Is the credit calendar-year or anniversary-year?');
  });
});

test('runQuestion returns structured benefit amounts for the calculation panel', async () => {
  const chunk = {
    id: 'travel-credit',
    heading: '$300 annual travel credit',
    text: 'Each account anniversary year.',
    score: 0.795,
    sourceUrl: 'https://creditcards.chase.com/rewards-credit-cards/sapphire/reserve',
    capturedAt: '2026-10-27',
  };
  const household = { card: { product: 'Chase Sapphire Reserve', issuer: 'Chase', benefitName: 'annual travel credit', benefitAmount: 300, usedAmount: 180 } };
  const result = await runQuestion('When does the credit apply?', {
    loadCorpus: async () => [chunk],
    loadSyntheticHousehold: async () => household,
    retrieve: async () => [chunk],
    answerQuestion: async ({ retrieveEnabled }) => ({ answer: retrieveEnabled ? 'grounded' : 'baseline', remainingAmount: 120, citations: [] }),
  });

  assert.equal(result.benefitAmount, 300);
  assert.equal(result.usedAmount, 180);
  assert.equal(result.remainingAmount, 120);
});

test('startServer binds only to the loopback interface', async () => {
  const server = await startServer({ port: 0, runQuestion: async () => comparison });
  try {
    assert.equal(server.address().address, '127.0.0.1');
  } finally {
    await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  }
});

test('rejects wrong content type, malformed JSON, missing questions, and long questions before running the agent', async () => {
  let calls = 0;
  const server = createServer({ runQuestion: async () => { calls += 1; return comparison; } });
  await usingServer(server, async (baseUrl) => {
    const wrongType = await fetch(`${baseUrl}/api/ask`, { method: 'POST', headers: { 'content-type': 'text/plain' }, body: '{}' });
    const malformed = await fetch(`${baseUrl}/api/ask`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: '{' });
    const missing = await fetch(`${baseUrl}/api/ask`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}' });
    const empty = await fetch(`${baseUrl}/api/ask`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ question: '  ' }) });
    const long = await fetch(`${baseUrl}/api/ask`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ question: 'q'.repeat(301) }) });

    assert.equal(wrongType.status, 415);
    assert.equal(malformed.status, 400);
    assert.equal(missing.status, 400);
    assert.equal(empty.status, 400);
    assert.equal(long.status, 400);
    assert.equal(calls, 0);
  });
});

test('rejects request bodies over 4 KiB and unknown/traversal paths', async () => {
  let calls = 0;
  const server = createServer({ runQuestion: async () => { calls += 1; return comparison; } });
  await usingServer(server, async (baseUrl) => {
    const oversized = await fetch(`${baseUrl}/api/ask`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ question: 'x'.repeat(5000) }),
    });
    const unknown = await fetch(`${baseUrl}/missing`);
    const traversal = await fetch(`${baseUrl}/%2e%2e/%2e%2e/etc/passwd`);

    assert.equal(oversized.status, 413);
    assert.equal(unknown.status, 404);
    assert.equal(traversal.status, 404);
    assert.equal(calls, 0);
  });
});

test('returns a user-readable local-model error without hosted fallback', async () => {
  const server = createServer({ runQuestion: async () => { throw new Error('Ollama answer request failed: local model unavailable'); } });
  await usingServer(server, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/ask`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ question: 'Ask something' }),
    });
    assert.equal(response.status, 503);
    assert.match((await response.json()).error, /local model unavailable/i);
  });
});

test('API preserves baseline and explicit no-evidence state with no citations', async () => {
  const household = { card: { product: 'Chase Sapphire Reserve', issuer: 'Chase' } };
  const server = createServer({
    runQuestion: (question) => runQuestion(question, {
      loadCorpus: async () => [],
      loadSyntheticHousehold: async () => household,
      retrieve: async () => [],
      answerQuestion: async ({ retrieveEnabled }) => retrieveEnabled
        ? { answer: 'I have insufficient evidence to answer that question.', remainingAmount: 120, citations: [] }
        : { answer: 'The synthetic record alone does not establish the cycle.', remainingAmount: 120, citations: [] },
    }),
  });

  await usingServer(server, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/ask`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ question: 'Is there a gasoline benefit?' }),
    });
    const result = await response.json();

    assert.equal(response.status, 200);
    assert.equal(result.status, 'insufficient-evidence');
    assert.match(result.baseline, /does not establish/i);
    assert.match(result.grounded, /insufficient evidence/i);
    assert.deepEqual(result.evidence, []);
    assert.deepEqual(result.citations, []);
  });
});
