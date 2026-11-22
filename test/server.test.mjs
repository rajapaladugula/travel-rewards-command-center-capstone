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

async function postAgents(baseUrl, body, contentType = 'application/json') {
  return fetch(`${baseUrl}/api/agents`, { method: 'POST', headers: { 'content-type': contentType }, body: typeof body === 'string' ? body : JSON.stringify(body) });
}

test('POST /api/agents runs the workflow with the validated mode and question', async () => {
  let received;
  const result = { mode: 'fact', status: 'answered', answer: 'ok', evidence: { facts: [], citations: [], missing_data: [] }, candidates: [], candidate_reviews: [], selected_candidate_id: null, retry_count: 0 };
  const server = createServer({ runAgentWorkflow: async (input) => { received = input; return result; } });
  await usingServer(server, async (baseUrl) => {
    const response = await postAgents(baseUrl, { question: ' How much remains? ', mode: 'fact' });
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), result);
    assert.deepEqual(received, { question: 'How much remains?', mode: 'fact' });
  });
});

test('POST /api/agents validates mode, question, content type, and method', async () => {
  const server = createServer({ runAgentWorkflow: async () => { throw new Error('must not run'); } });
  await usingServer(server, async (baseUrl) => {
    assert.equal((await postAgents(baseUrl, { question: 'q', mode: 'execute' })).status, 400);
    assert.equal((await postAgents(baseUrl, { question: 'q' })).status, 400);
    assert.equal((await postAgents(baseUrl, { question: '', mode: 'fact' })).status, 400);
    assert.equal((await postAgents(baseUrl, { question: 'x'.repeat(301), mode: 'fact' })).status, 400);
    assert.equal((await postAgents(baseUrl, '{"question":"q"}', 'text/plain')).status, 415);
    assert.equal((await fetch(`${baseUrl}/api/agents`)).status, 405);
  });
});

test('POST /api/agents allows one workflow at a time and then frees the slot', async () => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const result = { mode: 'fact', status: 'answered', answer: 'ok', evidence: {}, candidates: [], candidate_reviews: [], selected_candidate_id: null, retry_count: 0 };
  const server = createServer({ runAgentWorkflow: async () => { await gate; return result; } });
  await usingServer(server, async (baseUrl) => {
    const first = postAgents(baseUrl, { question: 'q', mode: 'fact' });
    await new Promise((resolve) => setTimeout(resolve, 50));
    assert.equal((await postAgents(baseUrl, { question: 'q', mode: 'fact' })).status, 429);
    release();
    assert.equal((await first).status, 200);
    assert.equal((await postAgents(baseUrl, { question: 'q', mode: 'fact' })).status, 200);
  });
});

test('POST /api/agents hides unexpected failure details and reports workflow unavailability', async () => {
  const unavailable = Object.assign(new Error('The multi-agent workflow is unavailable. Check that local Ollama is running.'), { statusCode: 503 });
  let error = unavailable;
  const server = createServer({ runAgentWorkflow: async () => { throw error; } });
  await usingServer(server, async (baseUrl) => {
    const first = await postAgents(baseUrl, { question: 'q', mode: 'fact' });
    assert.equal(first.status, 503);
    assert.match((await first.json()).error, /multi-agent workflow/i);
    error = new Error('SECRET /path');
    const second = await postAgents(baseUrl, { question: 'q', mode: 'fact' });
    assert.equal(second.status, 500);
    assert.doesNotMatch(JSON.stringify(await second.json()), /SECRET/);
  });
});

test('runQuestion labels a record-based answer instead of reporting insufficient evidence', async () => {
  const household = { card: { product: 'Chase Sapphire Reserve', issuer: 'Chase', benefitName: 'annual travel credit', benefitAmount: 300, usedAmount: 180 } };
  const result = await runQuestion('how much credit is left?', {
    loadCorpus: async () => [],
    loadSyntheticHousehold: async () => household,
    retrieve: async () => [],
    answerQuestion: async ({ retrieveEnabled }) => retrieveEnabled
      ? { answer: 'The synthetic record shows $120 remaining.', remainingAmount: 120, citations: [], source: 'household-record' }
      : { answer: 'baseline', remainingAmount: 120, citations: [] },
  });
  assert.equal(result.status, 'record-answer');
  assert.equal(result.remainingAmount, 120);
});

test('runQuestion labels help answers with their own status', async () => {
  const household = { card: { product: 'Chase Sapphire Reserve', issuer: 'Chase', benefitName: 'annual travel credit', benefitAmount: 300, usedAmount: 180 } };
  const result = await runQuestion('what can you help me with?', {
    loadCorpus: async () => [],
    loadSyntheticHousehold: async () => household,
    retrieve: async () => [],
    answerQuestion: async ({ retrieveEnabled }) => ({ answer: 'help text', remainingAmount: 120, citations: [], ...(retrieveEnabled ? { source: 'app-help' } : {}) }),
  });
  assert.equal(result.status, 'help');
});

test('runQuestion lists all of the card\'s saved sections for an overview question when retrieval finds none', async () => {
  const mk = (id, heading, product = 'Chase Sapphire Reserve') => ({ id, heading, text: heading, product, issuer: 'Chase', sourceUrl: 'https://example.com', capturedAt: '2026-10-27' });
  const chunks = [mk('a', 'Credit A'), mk('b', 'Credit B'), mk('c', 'Other card credit', 'Other Card')];
  const household = { card: { product: 'Chase Sapphire Reserve', issuer: 'Chase', benefitName: 'annual travel credit', benefitAmount: 300, usedAmount: 180 } };
  let groundedInput;
  const result = await runQuestion('what are the benefits of this card?', {
    loadCorpus: async () => chunks,
    loadSyntheticHousehold: async () => household,
    retrieve: async () => [],
    answerQuestion: async (input) => { if (input.retrieveEnabled) groundedInput = input; return { answer: 'overview', remainingAmount: 120, citations: [], ...(input.retrieveEnabled ? { source: 'benefit-overview' } : {}) }; },
  });
  assert.equal(result.status, 'overview');
  assert.equal(groundedInput.overview, true);
  assert.deepEqual(result.evidence.map((e) => e.id), ['a', 'b']);
});

test('runQuestion does not use the overview path for ordinary unsupported questions', async () => {
  const household = { card: { product: 'Chase Sapphire Reserve', issuer: 'Chase', benefitName: 'annual travel credit', benefitAmount: 300, usedAmount: 180 } };
  const result = await runQuestion('Does this card earn a bonus on gasoline purchases?', {
    loadCorpus: async () => [{ id: 'a', heading: 'A', text: 'A', product: 'Chase Sapphire Reserve', issuer: 'Chase' }],
    loadSyntheticHousehold: async () => household,
    retrieve: async () => [],
    answerQuestion: async () => ({ answer: 'insufficient evidence', remainingAmount: 120, citations: [] }),
  });
  assert.equal(result.status, 'insufficient-evidence');
  assert.deepEqual(result.evidence, []);
});

const routedHousehold = { card: { product: 'Chase Sapphire Reserve', issuer: 'Chase', benefitName: 'annual travel credit', benefitAmount: 300, usedAmount: 180, annualFee: 795, anniversaryDate: '08-15' } };
const routedChunks = ['A', 'B'].map((id) => ({ id, heading: `Credit ${id}`, text: `Credit ${id} text`, product: 'Chase Sapphire Reserve', issuer: 'Chase', sourceUrl: 'https://example.com', capturedAt: '2026-10-27' }));

test('help, record, and overview questions skip retrieval, judging, and rewriting entirely', async () => {
  for (const [question, status] of [['what can you help me with?', 'help'], ['how much credit is left?', 'record-answer'], ['what are the benefits of this card?', 'overview']]) {
    const calls = [];
    const result = await runQuestion(question, {
      loadCorpus: async () => routedChunks,
      loadSyntheticHousehold: async () => routedHousehold,
      retrieve: async () => { calls.push('retrieve'); return []; },
      judge: async () => { calls.push('judge'); return 'YES'; },
      rewrite: async () => { calls.push('rewrite'); return null; },
    });
    assert.equal(result.status, status, question);
    assert.deepEqual(calls, [], question);
  }
});

test('ordinary questions use verified retrieval: the judge decides which passages become evidence', async () => {
  const chunk = { ...routedChunks[0], score: 0.8 };
  const common = {
    loadCorpus: async () => routedChunks,
    loadSyntheticHousehold: async () => routedHousehold,
    retrieve: async () => [chunk],
    rewrite: async () => null,
    answerQuestion: async ({ retrieveEnabled, retrievedChunks }) => ({ answer: retrieveEnabled && retrievedChunks.length ? 'grounded text' : 'x', remainingAmount: 120, citations: [] }),
  };
  const accepted = await runQuestion('Tell me about credit A', { ...common, judge: async () => 'YES' });
  assert.equal(accepted.status, 'grounded');
  assert.deepEqual(accepted.evidence.map((e) => e.id), ['A']);

  const rejected = await runQuestion('Tell me about credit A', { ...common, judge: async () => 'NO' });
  assert.equal(rejected.status, 'insufficient-evidence');
  assert.deepEqual(rejected.evidence, []);
});
