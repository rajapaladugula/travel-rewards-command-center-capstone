import { spawn as nodeSpawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const MULTI_AGENT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '../multi_agent');
const MODES = new Set(['fact', 'plan']);
const STATUSES = new Set(['answered', 'abstained', 'clarification_required', 'failed']);
const DEFAULT_TIMEOUT_MS = 240_000;
const DEFAULT_MAX_OUTPUT_BYTES = 1_048_576;
const UNAVAILABLE = 'The multi-agent workflow is unavailable. Check that local Ollama is running and the Python environment is installed.';

function workflowError(message, statusCode = 503) {
  return Object.assign(new Error(message), { statusCode });
}

const asArray = (value) => (Array.isArray(value) ? value : []);

// Keep only what the page renders; the household record, prompts and raw model output never leave the process.
function projectResult(raw) {
  if (!raw || typeof raw !== 'object' || !MODES.has(raw.mode) || !STATUSES.has(raw.status)
    || typeof raw.answer !== 'string' || !raw.evidence || typeof raw.evidence !== 'object') {
    throw workflowError(UNAVAILABLE);
  }
  return {
    mode: raw.mode,
    status: raw.status,
    answer: raw.answer,
    evidence: {
      facts: asArray(raw.evidence.facts).map(({ id, label, value }) => ({ id, label, value })),
      citations: asArray(raw.evidence.citations).map(({ id, heading, source_url, captured_at, excerpt }) => ({ id, heading, source_url, captured_at, excerpt })),
      missing_data: asArray(raw.evidence.missing_data),
    },
    candidates: asArray(raw.candidates),
    candidate_reviews: asArray(raw.candidate_reviews),
    selected_candidate_id: typeof raw.selected_candidate_id === 'string' ? raw.selected_candidate_id : null,
    retry_count: Number.isInteger(raw.retry_count) ? raw.retry_count : 0,
  };
}

export function runAgentWorkflow(
  { mode, question },
  { spawn = nodeSpawn, timeoutMs = DEFAULT_TIMEOUT_MS, maxOutputBytes = DEFAULT_MAX_OUTPUT_BYTES, projectDir = MULTI_AGENT_DIR } = {},
) {
  if (!MODES.has(mode)) return Promise.reject(workflowError('Mode must be "fact" or "plan"', 400));

  return new Promise((resolveRun, reject) => {
    let settled = false;
    let size = 0;
    const chunks = [];
    const finish = (callback) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      callback();
    };

    const args = ['run', '--python', '3.12', '--project', projectDir, 'travel-rewards-agent', '--mode', mode, '--question', question];
    const child = spawn('uv', args, { stdio: ['ignore', 'pipe', 'pipe'], env: process.env, shell: false });
    const timer = setTimeout(() => {
      finish(() => reject(workflowError('The multi-agent workflow timed out. Try a shorter question or run it again.')));
      child.kill('SIGKILL');
    }, timeoutMs);

    child.stdout.on('data', (chunk) => {
      size += chunk.length;
      if (size > maxOutputBytes) {
        finish(() => reject(workflowError(UNAVAILABLE)));
        child.kill('SIGKILL');
      } else {
        chunks.push(chunk);
      }
    });
    child.stderr.resume(); // diagnostics are intentionally discarded, never returned to the browser
    child.on('error', () => finish(() => reject(workflowError(UNAVAILABLE))));
    child.on('close', () => finish(() => {
      try {
        resolveRun(projectResult(JSON.parse(Buffer.concat(chunks).toString('utf8'))));
      } catch (error) {
        reject(error.statusCode ? error : workflowError(UNAVAILABLE));
      }
    }));
  });
}
