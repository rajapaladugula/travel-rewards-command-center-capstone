import test from 'node:test';
import assert from 'node:assert/strict';
import { embedTexts } from '../src/embeddings.mjs';
import { retrieve } from '../src/retriever.mjs';

const baseChunk = { issuer: 'Chase', product: 'Chase Sapphire Reserve' };

function chunksWithVectors() {
  return [
    { ...baseChunk, id: 'travel', heading: 'Travel credit', text: 'Travel credit resets each account anniversary year.', vector: [0.98, 0.2] },
    { ...baseChunk, id: 'lounges', heading: 'Lounge access', text: 'Visit airport lounges with eligible guests.', vector: [0.1, 0.99] },
    { ...baseChunk, id: 'hotel', heading: 'Hotel credit', text: 'Prepaid hotel booking benefit.', vector: [0.8, 0.6] },
    { ...baseChunk, id: 'lyft', heading: 'Ride credits', text: 'Monthly rideshare credit.', vector: [0.7, 0.7] },
    { issuer: 'Other Bank', product: 'Other Travel Card', id: 'foreign', heading: 'Travel credit', text: 'Travel credit resets on January 1.', vector: [1, 0] },
  ];
}

test('sends a batch to Ollama embeddings and validates the returned vectors', async () => {
  let request;
  const result = await embedTexts(['query text', 'document text'], {
    baseUrl: 'http://localhost:11434',
    fetchImpl: async (url, options) => {
      request = { url, options };
      return new Response(JSON.stringify({ embeddings: [[1, 0], [0.5, 0.5]] }), { status: 200 });
    },
  });

  assert.equal(request.url, 'http://localhost:11434/api/embed');
  assert.deepEqual(JSON.parse(request.options.body), { model: 'nomic-embed-text', input: ['query text', 'document text'] });
  assert.deepEqual(result, [[1, 0], [0.5, 0.5]]);
});

test('reports a clear error when Ollama cannot provide embeddings', async () => {
  await assert.rejects(
    embedTexts(['query'], { fetchImpl: async () => { throw new Error('connection refused'); } }),
    /Ollama embedding request failed.*connection refused/i,
  );
  await assert.rejects(
    embedTexts(['query'], { fetchImpl: async () => new Response(JSON.stringify({ embeddings: [] }), { status: 200 }) }),
    /invalid embeddings response/i,
  );
});

test('ranks semantically closest chunks, filters other products, and returns at most topK', async () => {
  const chunks = chunksWithVectors();
  const embed = async (texts) => texts.map((text) => {
    if (text === 'when does the travel credit reset?') return [1, 0];
    return chunks.find((chunk) => chunk.text === text).vector;
  });

  const results = await retrieve({
    query: 'when does the travel credit reset?',
    chunks,
    product: 'Chase Sapphire Reserve',
    issuer: 'Chase',
    topK: 3,
    minScore: 0.5,
    embed,
  });

  assert.deepEqual(results.map((chunk) => chunk.id), ['travel', 'hotel', 'lyft']);
  assert.ok(results.every((chunk) => chunk.score >= 0.5));
  assert.equal(results.length, 3);
});

test('default relevance threshold removes semantically adjacent but off-topic passages', async () => {
  const makeVector = (cosine) => [cosine, Math.sqrt(1 - cosine * cosine)];
  const chunks = [
    { ...baseChunk, id: 'direct', text: 'Direct annual-credit rule.', vector: makeVector(0.8) },
    { ...baseChunk, id: 'adjacent', text: 'A different annual hotel benefit.', vector: makeVector(0.65) },
  ];
  const results = await retrieve({
    query: 'calendar or account anniversary travel credit',
    chunks,
    product: 'Chase Sapphire Reserve',
    issuer: 'Chase',
    embed: async (texts) => texts.map((text) => text === 'calendar or account anniversary travel credit'
      ? [1, 0]
      : chunks.find((chunk) => chunk.text === text).vector),
  });

  assert.deepEqual(results.map((chunk) => chunk.id), ['direct']);
});

test('returns no results below the configured relevance threshold', async () => {
  const chunks = [{ ...baseChunk, id: 'lounges', text: 'Airport lounge access.', vector: [0, 1] }];
  const results = await retrieve({
    query: 'when does the travel credit reset?',
    chunks,
    product: 'Chase Sapphire Reserve',
    issuer: 'Chase',
    minScore: 0.8,
    embed: async (texts) => texts.map((text) => text === 'when does the travel credit reset?' ? [1, 0] : [0, 1]),
  });

  assert.deepEqual(results, []);
});

test('rejects mismatched embedding dimensions', async () => {
  await assert.rejects(
    retrieve({
      query: 'query',
      chunks: [{ ...baseChunk, id: 'a', text: 'benefit', vector: [1, 0, 0] }],
      product: 'Chase Sapphire Reserve',
      issuer: 'Chase',
      embed: async () => [[1, 0], [1, 0, 0]],
    }),
    /embedding dimensions must match/i,
  );
});
