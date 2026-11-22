import assert from 'node:assert/strict';
import test from 'node:test';
import { generateWithOllama } from '../src/agent.mjs';
import { embedTexts } from '../src/embeddings.mjs';
import { validateOllamaBaseUrl } from '../src/local-ollama.mjs';

test('normalizes supported loopback Ollama origins', () => {
  assert.equal(validateOllamaBaseUrl('http://localhost:11434'), 'http://localhost:11434');
  assert.equal(validateOllamaBaseUrl('http://127.0.0.1:11434/'), 'http://127.0.0.1:11434');
  assert.equal(validateOllamaBaseUrl('http://[::1]:11434'), 'http://[::1]:11434');
});

test('rejects non-loopback or non-origin Ollama endpoints', () => {
  for (const baseUrl of [
    'https://localhost:11434',
    'http://ollama.example:11434',
    'http://203.0.113.7:11434',
    'http://localhost:11434/proxy',
    'http://user:password@localhost:11434',
    'http://localhost:11434?target=remote',
    'http://localhost:11434#remote',
    'not a URL',
  ]) {
    assert.throws(() => validateOllamaBaseUrl(baseUrl), /OLLAMA_HOST must be an HTTP loopback origin/i);
  }
});

test('embedding client rejects a remote endpoint before sending source text', async () => {
  let requested = false;

  await assert.rejects(
    embedTexts(['synthetic source text'], {
      baseUrl: 'http://ollama.example:11434',
      fetchImpl: async () => { requested = true; },
    }),
    /OLLAMA_HOST must be an HTTP loopback origin/i,
  );
  assert.equal(requested, false);
});

test('answer client rejects a remote endpoint before sending the question', async () => {
  let requested = false;

  await assert.rejects(
    generateWithOllama('synthetic question', {
      baseUrl: 'http://ollama.example:11434',
      fetchImpl: async () => { requested = true; },
    }),
    /OLLAMA_HOST must be an HTTP loopback origin/i,
  );
  assert.equal(requested, false);
});

import { judgeWithOllama, rewriteWithOllama } from '../src/local-judge.mjs';

test('judge and rewrite calls stay on loopback and send a one-word-answer prompt', async () => {
  const requests = [];
  const fetchImpl = async (url, init) => { requests.push({ url, body: JSON.parse(init.body) }); return { ok: true, json: async () => ({ message: { content: ' YES ' } }) }; };
  assert.equal(await judgeWithOllama('How many guests?', { heading: 'Lounge access', text: 'Up to two guests.' }, { fetchImpl }), 'YES');
  assert.equal(await rewriteWithOllama('when does it end', { fetchImpl }), 'YES');
  assert.ok(requests.every((r) => r.url.startsWith('http://localhost:11434/')));
  assert.equal(requests[0].body.options.temperature, 0);
  assert.match(requests[0].body.messages[0].content, /exactly one word: YES or NO/);
  assert.match(requests[0].body.messages[1].content, /Question: How many guests\?/);
  await assert.rejects(() => judgeWithOllama('q', { heading: 'h', text: 't' }, { baseUrl: 'https://remote.example.com', fetchImpl }), /loopback|local/i);
});

import { selectWithOllama } from '../src/local-judge.mjs';

test('selection call numbers the passages and asks for a number or NONE', async () => {
  let body;
  const fetchImpl = async (_url, init) => { body = JSON.parse(init.body); return { ok: true, json: async () => ({ message: { content: '2' } }) }; };
  const answer = await selectWithOllama('How many guests?', [{ heading: 'A', text: 'ta' }, { heading: 'B', text: 'tb' }], { fetchImpl });
  assert.equal(answer, '2');
  assert.match(body.messages[0].content, /only the passage number or the word NONE/);
  assert.match(body.messages[1].content, /Passage 1 \(A\): ta/);
  assert.match(body.messages[1].content, /Passage 2 \(B\): tb/);
  assert.equal(body.options.temperature, 0);
});
