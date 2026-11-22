import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { runAgentWorkflow } from '../src/agent-bridge.mjs';

const workflowResult = {
  mode: 'plan',
  status: 'clarification_required',
  answer: 'No plan met the safety gate.',
  evidence: {
    goal: 'g',
    household: { card_name: 'secret-household-field' },
    facts: [{ id: 'f1', label: 'Remaining', value: 120 }],
    citations: [{ id: 'c1', heading: 'Credit', source_url: 'https://example.com/a', captured_at: '2026-10-27', excerpt: 'Anniversary year.' }],
    missing_data: [],
    direct_answer: null,
    answer_evidence_ids: [],
  },
  candidates: [],
  candidate_reviews: [{ candidate_id: 'p1', scores: { grounding: 1, goal_fit: 1, feasibility: 1, clarity: 1 }, hard_failures: ['x'], feedback: [] }],
  selected_candidate_id: null,
  retry_count: 1,
};

function fakeSpawn({ stdout = '', stderr = '', code = 0, hang = false, error } = {}) {
  const calls = [];
  const spawn = (command, args, options) => {
    calls.push({ command, args, options });
    const child = new EventEmitter();
    child.stdout = new PassThrough();
    child.stderr = new PassThrough();
    child.killed = false;
    child.kill = () => { child.killed = true; child.emit('close', null); };
    queueMicrotask(() => {
      if (error) { child.emit('error', error); return; }
      if (hang) return;
      child.stdout.end(stdout);
      child.stderr.end(stderr);
      child.emit('close', code);
    });
    return child;
  };
  spawn.calls = calls;
  return spawn;
}

test('runs the Python CLI with an argument array and no shell', async () => {
  const spawn = fakeSpawn({ stdout: JSON.stringify(workflowResult) });
  await runAgentWorkflow({ mode: 'plan', question: 'a; rm -rf / $(x)' }, { spawn });
  const { command, args, options } = spawn.calls[0];
  assert.equal(command, 'uv');
  assert.deepEqual(args.slice(0, 5), ['run', '--python', '3.12', '--project', args[4]]);
  assert.ok(args[4].endsWith('multi_agent'));
  assert.deepEqual(args.slice(5), ['travel-rewards-agent', '--mode', 'plan', '--question', 'a; rm -rf / $(x)']);
  assert.notEqual(options.shell, true);
});

test('returns only the fields the page renders and drops the household record', async () => {
  const result = await runAgentWorkflow({ mode: 'plan', question: 'q' }, { spawn: fakeSpawn({ stdout: JSON.stringify(workflowResult) }) });
  assert.equal(result.status, 'clarification_required');
  assert.equal(result.selected_candidate_id, null);
  assert.equal(result.retry_count, 1);
  assert.equal(result.evidence.citations[0].source_url, 'https://example.com/a');
  assert.equal(result.evidence.household, undefined);
  assert.equal(result.candidate_reviews[0].hard_failures[0], 'x');
});

test('a failed workflow (exit code 1 with valid JSON) is returned, not thrown', async () => {
  const failed = { ...workflowResult, status: 'failed' };
  const result = await runAgentWorkflow({ mode: 'plan', question: 'q' }, { spawn: fakeSpawn({ stdout: JSON.stringify(failed), code: 1 }) });
  assert.equal(result.status, 'failed');
});

test('rejects an unknown mode before spawning', async () => {
  const spawn = fakeSpawn();
  await assert.rejects(() => runAgentWorkflow({ mode: 'execute', question: 'q' }, { spawn }), /mode/i);
  assert.equal(spawn.calls.length, 0);
});

test('non-JSON, wrong-shape, and nonzero-without-JSON output use fixed messages without stderr', async () => {
  for (const options of [
    { stdout: 'not json', stderr: 'SECRET traceback' },
    { stdout: JSON.stringify({ status: 'answered' }), stderr: 'SECRET traceback' },
    { stdout: '', stderr: 'SECRET traceback', code: 1 },
  ]) {
    await assert.rejects(
      () => runAgentWorkflow({ mode: 'fact', question: 'q' }, { spawn: fakeSpawn(options) }),
      (error) => { assert.doesNotMatch(error.message, /SECRET/); assert.match(error.message, /multi-agent workflow/i); return true; },
    );
  }
});

test('times out and kills a hung workflow', async () => {
  const spawn = fakeSpawn({ hang: true });
  await assert.rejects(() => runAgentWorkflow({ mode: 'fact', question: 'q' }, { spawn, timeoutMs: 20 }), /timed out/i);
});

test('maps a spawn failure to a fixed message', async () => {
  await assert.rejects(
    () => runAgentWorkflow({ mode: 'fact', question: 'q' }, { spawn: fakeSpawn({ error: new Error('ENOENT /secret/path') }) }),
    (error) => { assert.doesNotMatch(error.message, /secret/); return true; },
  );
});

test('rejects output over the size cap', async () => {
  await assert.rejects(
    () => runAgentWorkflow({ mode: 'fact', question: 'q' }, { spawn: fakeSpawn({ stdout: 'x'.repeat(2000) }), maxOutputBytes: 1000 }),
    /multi-agent workflow/i,
  );
});
