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
