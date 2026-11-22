import { createServer as createHttpServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { loadCorpus, loadSyntheticHousehold } from './corpus.mjs';
import { retrieve } from './retriever.mjs';
import { answerQuestion, classifyQuestion } from './agent.mjs';
import { verifiedRetrieve } from './verified-retrieval.mjs';
import { judgeWithOllama, rewriteWithOllama } from './local-judge.mjs';
import { runAgentWorkflow as defaultRunAgentWorkflow } from './agent-bridge.mjs';

const HOST = '127.0.0.1';
const DEFAULT_PORT = 4173;
const MAX_BODY_BYTES = 4096;
const MAX_QUESTION_LENGTH = 300;
const PUBLIC_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '../public');
const ASSETS = new Map([
  ['/', ['index.html', 'text/html; charset=utf-8']],
  ['/app.js', ['app.js', 'text/javascript; charset=utf-8']],
  ['/styles.css', ['styles.css', 'text/css; charset=utf-8']],
]);

function sendJson(response, status, body) {
  response.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
    'x-content-type-options': 'nosniff',
  });
  response.end(JSON.stringify(body));
}

function readRequestBody(request) {
  return new Promise((resolveBody, reject) => {
    const declaredLength = Number(request.headers['content-length'] ?? 0);
    if (declaredLength > MAX_BODY_BYTES) {
      request.resume();
      reject(Object.assign(new Error('Request body exceeds 4096 bytes'), { statusCode: 413 }));
      return;
    }

    const chunks = [];
    let size = 0;
    let tooLarge = false;
    request.on('data', (chunk) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        tooLarge = true;
        chunks.length = 0;
      } else if (!tooLarge) {
        chunks.push(chunk);
      }
    });
    request.on('end', () => {
      if (tooLarge) {
        reject(Object.assign(new Error('Request body exceeds 4096 bytes'), { statusCode: 413 }));
        return;
      }
      try {
        resolveBody(JSON.parse(Buffer.concat(chunks).toString('utf8')));
      } catch {
        reject(Object.assign(new Error('Request body must be valid JSON'), { statusCode: 400 }));
      }
    });
    request.on('error', reject);
  });
}

export async function runQuestion(question, deps = {}) {
  const loadCorpusFn = deps.loadCorpus ?? loadCorpus;
  const loadHouseholdFn = deps.loadSyntheticHousehold ?? loadSyntheticHousehold;
  const retrieveFn = deps.retrieve ?? retrieve;
  const answerFn = deps.answerQuestion ?? answerQuestion;
  const judge = deps.judge ?? judgeWithOllama;
  const rewrite = deps.rewrite ?? rewriteWithOllama;
  const [chunks, household] = await Promise.all([loadCorpusFn(), loadHouseholdFn()]);
  const card = household.card;
  const baseline = await answerFn({ question, household, retrieveEnabled: false });

  // Help, record, and overview questions are answered by code. Everything else goes through verified retrieval:
  // a similarity search, a deterministic distinctive-word gate, a strict model check, and at most one query rewrite.
  const route = classifyQuestion(question, card);
  let evidence = [];
  if (route === 'overview') evidence = chunks.filter((chunk) => chunk.product === card.product && chunk.issuer === card.issuer);
  else if (route === null) {
    ({ results: evidence } = await verifiedRetrieve({ question, chunks, card, retrieveFn, judge, rewrite }));
  }
  const overview = route === 'overview';
  const grounded = await answerFn({ question, household, retrievedChunks: evidence, retrieveEnabled: true, overview });

  return {
    status: grounded.source === 'benefit-overview' ? 'overview' : evidence.length ? 'grounded' : grounded.source === 'household-record' ? 'record-answer' : grounded.source === 'app-help' ? 'help' : 'insufficient-evidence',
    baseline: baseline.answer,
    grounded: grounded.answer,
    benefitAmount: card.benefitAmount,
    usedAmount: card.usedAmount,
    remainingAmount: grounded.remainingAmount,
    evidence: evidence.map(({ id, heading, text, score, sourceUrl, capturedAt }) => ({ id, heading, text, score, sourceUrl, capturedAt })),
    citations: grounded.citations,
  };
}

export function createServer({ runQuestion: runQuestionFn = runQuestion, runAgentWorkflow: runAgentWorkflowFn = defaultRunAgentWorkflow, publicDir = PUBLIC_DIR } = {}) {
  let agentRunActive = false;
  return createHttpServer(async (request, response) => {
    const url = new URL(request.url ?? '/', 'http://127.0.0.1');
    const asset = ASSETS.get(url.pathname);

    if (request.method === 'GET' && asset) {
      try {
        const content = await readFile(resolve(publicDir, asset[0]));
        response.writeHead(200, {
          'content-type': asset[1],
          'cache-control': 'no-store',
          'x-content-type-options': 'nosniff',
        });
        response.end(content);
      } catch {
        sendJson(response, 404, { error: 'Not found' });
      }
      return;
    }

    if (url.pathname !== '/api/ask' && url.pathname !== '/api/agents') {
      sendJson(response, 404, { error: 'Not found' });
      return;
    }
    if (request.method !== 'POST') {
      sendJson(response, 405, { error: 'Use POST for this endpoint' });
      return;
    }
    if ((request.headers['content-type'] ?? '').split(';')[0].trim().toLowerCase() !== 'application/json') {
      sendJson(response, 415, { error: 'Content-Type must be application/json' });
      return;
    }

    let body;
    try {
      body = await readRequestBody(request);
    } catch (error) {
      sendJson(response, error.statusCode ?? 400, { error: error.message });
      return;
    }
    if (typeof body?.question !== 'string' || !body.question.trim()) {
      sendJson(response, 400, { error: 'Question must be a non-empty string' });
      return;
    }
    if (body.question.length > MAX_QUESTION_LENGTH) {
      sendJson(response, 400, { error: `Question must be ${MAX_QUESTION_LENGTH} characters or fewer` });
      return;
    }

    if (url.pathname === '/api/agents') {
      if (body.mode !== 'fact' && body.mode !== 'plan') {
        sendJson(response, 400, { error: 'Mode must be "fact" or "plan"' });
        return;
      }
      if (agentRunActive) {
        sendJson(response, 429, { error: 'A multi-agent workflow is already running. Wait for it to finish.' });
        return;
      }
      agentRunActive = true;
      try {
        sendJson(response, 200, await runAgentWorkflowFn({ question: body.question.trim(), mode: body.mode }));
      } catch (error) {
        const known = error?.statusCode === 503 || error?.statusCode === 400;
        sendJson(response, known ? error.statusCode : 500, {
          error: known ? error.message : 'Unable to process the multi-agent request.',
        });
      } finally {
        agentRunActive = false;
      }
      return;
    }

    try {
      sendJson(response, 200, await runQuestionFn(body.question.trim()));
    } catch (error) {
      const localModelFailure = /ollama|model.*unavailable|model.*not found/i.test(error.message);
      sendJson(response, localModelFailure ? 503 : 500, {
        error: localModelFailure ? error.message : 'Unable to process the local assistant request.',
      });
    }
  });
}

export async function startServer({ port = Number(process.env.PORT ?? DEFAULT_PORT), runQuestion: runQuestionFn } = {}) {
  if (!Number.isInteger(port) || port < 0 || port > 65535) throw new Error('PORT must be between 0 and 65535');
  const server = createServer({ runQuestion: runQuestionFn });
  await new Promise((resolveListen, reject) => {
    server.once('error', reject);
    server.listen(port, HOST, resolveListen);
  });
  return server;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  startServer().then((server) => {
    console.log(`Travel Rewards RAG UI available at http://${HOST}:${server.address().port}`);
  }).catch((error) => {
    console.error(`Could not start the local UI: ${error.message}`);
    process.exitCode = 1;
  });
}
