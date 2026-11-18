import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import { calculateRemaining } from './agent.mjs';
import { loadCorpus, loadSyntheticHousehold } from './corpus.mjs';
import { retrieve } from './retriever.mjs';

const MAX_QUERY_LENGTH = 2000;
const SAFE_TOOL_ERRORS = new Set([
  'Search query must be non-empty',
  `Search query must be at most ${MAX_QUERY_LENGTH} characters`,
]);

const TOOL_DEFINITIONS = [
  {
    name: 'get_household_snapshot',
    description: 'Read the synthetic card and benefit fields used by this demo.',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
  },
  {
    name: 'search_benefit_evidence',
    description: 'Search the dated public benefit snapshot for relevant evidence.',
    inputSchema: {
      type: 'object',
      properties: { query: { type: 'string', minLength: 1, maxLength: MAX_QUERY_LENGTH } },
      required: ['query'],
      additionalProperties: false,
    },
  },
  {
    name: 'calculate_remaining_benefit',
    description: 'Calculate the remaining synthetic benefit using deterministic arithmetic.',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
  },
];

function pickCardFields(household) {
  const card = household?.card;
  if (!card) throw new Error('Synthetic household card is required');
  return {
    product: card.product,
    issuer: card.issuer,
    annualFee: card.annualFee,
    benefitName: card.benefitName,
    benefitAmount: card.benefitAmount,
    usedAmount: card.usedAmount,
    anniversaryDate: card.anniversaryDate,
  };
}

function pickEvidenceFields(chunk) {
  return {
    id: chunk.id,
    heading: chunk.heading,
    text: chunk.text,
    score: chunk.score,
    sourceUrl: chunk.sourceUrl,
    capturedAt: chunk.capturedAt,
  };
}

export function createToolHandlers({
  loadCorpus: loadCorpusImpl = loadCorpus,
  loadHousehold: loadHouseholdImpl = loadSyntheticHousehold,
  retrieve: retrieveImpl = retrieve,
  calculateRemaining: calculateRemainingImpl = calculateRemaining,
} = {}) {
  return {
    async get_household_snapshot() {
      const household = await loadHouseholdImpl();
      return pickCardFields(household);
    },

    async search_benefit_evidence(args = {}) {
      const query = args?.query;
      if (typeof query !== 'string' || !query.trim()) throw new Error('Search query must be non-empty');
      const normalizedQuery = query.trim();
      if (normalizedQuery.length > MAX_QUERY_LENGTH) throw new Error(`Search query must be at most ${MAX_QUERY_LENGTH} characters`);

      const [chunks, household] = await Promise.all([loadCorpusImpl(), loadHouseholdImpl()]);
      const card = pickCardFields(household);
      const matches = await retrieveImpl({
        query: normalizedQuery,
        chunks,
        product: card.product,
        issuer: card.issuer,
        topK: 3,
        minScore: 0.7,
      });
      if (!Array.isArray(matches)) throw new Error('Retriever returned an invalid result list');

      return {
        matches: matches
          .filter((chunk) => Number.isFinite(chunk.score) && chunk.score >= 0.7)
          .slice(0, 3)
          .map(pickEvidenceFields),
      };
    },

    async calculate_remaining_benefit() {
      const household = await loadHouseholdImpl();
      const card = pickCardFields(household);
      const remainingAmount = calculateRemainingImpl({
        benefitAmount: card.benefitAmount,
        usedAmount: card.usedAmount,
      });
      return {
        benefitName: card.benefitName,
        benefitAmount: card.benefitAmount,
        usedAmount: card.usedAmount,
        remainingAmount,
        anniversaryDate: card.anniversaryDate,
        currency: 'USD',
      };
    },
  };
}

export function getToolDefinitions() {
  return structuredClone(TOOL_DEFINITIONS);
}

export function formatToolError(error) {
  const message = error instanceof Error ? error.message : '';
  if (SAFE_TOOL_ERRORS.has(message)) return message;
  if (message.startsWith('Unknown MCP tool: ')) return 'Unknown MCP tool';
  return 'Unable to process this tool request safely.';
}

export async function callTool(handlers, name, args = {}) {
  if (!Object.hasOwn(handlers, name)) throw new Error(`Unknown MCP tool: ${name}`);
  return handlers[name](args);
}

export async function startMcpServer({ handlers = createToolHandlers(), transport = new StdioServerTransport() } = {}) {
  const server = new Server(
    { name: 'travel-rewards-capstone-tools', version: '1.0.0' },
    { capabilities: { tools: {} } },
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: getToolDefinitions() }));
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    try {
      const result = await callTool(handlers, request.params.name, request.params.arguments ?? {});
      return { content: [{ type: 'text', text: JSON.stringify(result) }] };
    } catch (error) {
      return { content: [{ type: 'text', text: formatToolError(error) }], isError: true };
    }
  });

  await server.connect(transport);
  return server;
}

const invokedPath = process.argv[1] ? pathToFileURL(resolve(process.argv[1])).href : null;
if (import.meta.url === invokedPath) {
  startMcpServer().catch((error) => {
    console.error(`Travel Rewards MCP server failed: ${formatToolError(error)}`);
    process.exitCode = 1;
  });
}
