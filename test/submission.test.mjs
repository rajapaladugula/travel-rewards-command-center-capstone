import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const submissionPath = resolve(dirname(fileURLToPath(import.meta.url)), '../submission.md');

test('submission is 600–900 words and covers retrieval evidence and design', async () => {
  const submission = await readFile(submissionPath, 'utf8');
  const words = submission.trim().split(/\s+/);

  assert.ok(words.length >= 600 && words.length <= 900, `expected 600–900 words, got ${words.length}`);
  for (const [label, pattern] of [
    ['retrieval decision', /retrieval is required/i],
    ['public source', /https:\/\/creditcards\.chase\.com/i],
    ['chunking', /chunk/i],
    ['top-three retrieval', /top three/i],
    ['no-retrieval comparison', /without retrieval/i],
    ['retrieval-grounded comparison', /with retrieval/i],
    ['failure mitigation', /stale|irrelevant|failure/i],
    ['abstention', /abstain|insufficient evidence/i],
    ['synthetic data', /synthetic/i],
  ]) {
    assert.match(submission, pattern, `missing ${label}`);
  }
});
