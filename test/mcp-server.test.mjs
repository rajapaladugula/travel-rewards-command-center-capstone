import assert from 'node:assert/strict';
import test from 'node:test';
import { callTool, createToolHandlers, formatToolError, getToolDefinitions } from '../src/mcp-server.mjs';

const householdFixture = Object.freeze({
  householdId: 'synthetic-household-secret-id',
  member: Object.freeze({ memberId: 'synthetic-member-secret-id', name: 'Sample Traveler' }),
  card: Object.freeze({
    cardId: 'synthetic-card-secret-id',
    product: 'Chase Sapphire Reserve',
    issuer: 'Chase',
    annualFee: 795,
    benefitName: 'annual travel credit',
    benefitAmount: 300,
    usedAmount: 180,
    anniversaryDate: '08-15',
  }),
});

const publicChunks = [
  {
    id: 'travel-credit-1',
    product: 'Chase Sapphire Reserve',
    issuer: 'Chase',
    heading: 'Travel Credit',
    text: 'The travel credit applies during the account anniversary year.',
    sourceUrl: 'https://creditcards.chase.com/rewards-credit-cards/sapphire/reserve',
    capturedAt: '2026-10-27',
  },
];

async function makeHandlers(overrides = {}) {
  return createToolHandlers({
    loadCorpus: async () => publicChunks,
    loadHousehold: async () => householdFixture,
    retrieve: async () => [],
    calculateRemaining: ({ benefitAmount, usedAmount }) => benefitAmount - usedAmount,
    ...overrides,
  });
}

test('exposes exactly the three read-only travel rewards tools', async () => {
  const handlers = await makeHandlers();

  assert.deepEqual(Object.keys(handlers).sort(), [
    'calculate_remaining_benefit',
    'get_household_snapshot',
    'search_benefit_evidence',
  ]);
});

test('MCP tool definitions expose exactly the three approved capabilities', () => {
  const definitions = getToolDefinitions();

  assert.deepEqual(definitions.map(({ name }) => name).sort(), [
    'calculate_remaining_benefit',
    'get_household_snapshot',
    'search_benefit_evidence',
  ]);
  assert.deepEqual(definitions.find(({ name }) => name === 'search_benefit_evidence').inputSchema.required, ['query']);
  assert.equal(definitions.every(({ inputSchema }) => inputSchema.additionalProperties === false), true);
});

test('household tool returns only the synthetic card fields required by agents', async () => {
  const handlers = await makeHandlers();
  const before = JSON.stringify(householdFixture);

  const result = await handlers.get_household_snapshot({});

  assert.deepEqual(result, {
    product: 'Chase Sapphire Reserve',
    issuer: 'Chase',
    annualFee: 795,
    benefitName: 'annual travel credit',
    benefitAmount: 300,
    usedAmount: 180,
    anniversaryDate: '08-15',
  });
  assert.equal(JSON.stringify(householdFixture), before);
  assert.equal(JSON.stringify(result).includes('synthetic-household-secret-id'), false);
  assert.equal(JSON.stringify(result).includes('Sample Traveler'), false);
});

test('retrieval applies the fixture card filter and the approved top-three threshold', async () => {
  let request;
  const handlers = await makeHandlers({
    retrieve: async (input) => {
      request = input;
      return [{ ...publicChunks[0], score: 0.82 }];
    },
  });

  const result = await handlers.search_benefit_evidence({ query: 'remaining annual travel credit' });

  assert.equal(request.query, 'remaining annual travel credit');
  assert.deepEqual(request.chunks, publicChunks);
  assert.equal(request.product, 'Chase Sapphire Reserve');
  assert.equal(request.issuer, 'Chase');
  assert.equal(request.topK, 3);
  assert.equal(request.minScore, 0.7);
  assert.deepEqual(result.matches, [{
    id: 'travel-credit-1',
    heading: 'Travel Credit',
    text: 'The travel credit applies during the account anniversary year.',
    score: 0.82,
    sourceUrl: 'https://creditcards.chase.com/rewards-credit-cards/sapphire/reserve',
    capturedAt: '2026-10-27',
  }]);
});

test('retrieval returns no unsupported matches when the retriever finds no relevant passage', async () => {
  const handlers = await makeHandlers({ retrieve: async () => [] });

  const result = await handlers.search_benefit_evidence({ query: 'unrelated gasoline benefit' });

  assert.deepEqual(result, { matches: [] });
});

test('calculator returns the deterministic remaining benefit without changing the fixture', async () => {
  const handlers = await makeHandlers();
  const before = JSON.stringify(householdFixture);

  const result = await handlers.calculate_remaining_benefit({});

  assert.deepEqual(result, {
    benefitName: 'annual travel credit',
    benefitAmount: 300,
    usedAmount: 180,
    remainingAmount: 120,
    anniversaryDate: '08-15',
    currency: 'USD',
  });
  assert.equal(JSON.stringify(householdFixture), before);
});

test('search rejects an empty query and dispatch rejects an unknown tool name', async () => {
  const handlers = await makeHandlers();

  await assert.rejects(handlers.search_benefit_evidence({ query: '  ' }), /query must be non-empty/i);
  await assert.rejects(callTool(handlers, 'make_payment', {}), /unknown MCP tool/i);
});

test('MCP error formatter preserves allow-listed input errors and hides unexpected details', () => {
  assert.equal(formatToolError(new Error('Search query must be non-empty')), 'Search query must be non-empty');
  assert.equal(formatToolError(new Error('Unknown MCP tool: make_payment')), 'Unknown MCP tool');
  const safe = formatToolError(new Error(`secret-token /private/path ${'stack '.repeat(100)}`));
  assert.equal(safe.length <= 240, true);
  assert.equal(safe.includes('secret-token'), false);
  assert.equal(safe.includes('/private/path'), false);
});
