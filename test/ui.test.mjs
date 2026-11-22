import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from '../src/server.mjs';

async function withServer(callback) {
  const server = createServer({ runQuestion: async () => ({ baseline: '', grounded: '', remainingAmount: 0, evidence: [], citations: [] }) });
  await new Promise((resolve, reject) => server.once('error', reject).listen(0, '127.0.0.1', resolve));
  try {
    return await callback(`http://127.0.0.1:${server.address().port}`);
  } finally {
    await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  }
}

test('serves the allowlisted HTML, JavaScript, and stylesheet with correct content types', async () => {
  await withServer(async (baseUrl) => {
    const page = await fetch(baseUrl);
    const script = await fetch(`${baseUrl}/app.js`);
    const styles = await fetch(`${baseUrl}/styles.css`);

    assert.equal(page.status, 200);
    assert.match(page.headers.get('content-type'), /text\/html/i);
    assert.match(await page.text(), /id="question-form"/);
    assert.equal(script.status, 200);
    assert.match(script.headers.get('content-type'), /javascript/i);
    assert.equal(styles.status, 200);
    assert.match(styles.headers.get('content-type'), /text\/css/i);
  });
});

test('page exposes accessible question and result regions', async () => {
  await withServer(async (baseUrl) => {
    const html = await (await fetch(baseUrl)).text();
    const script = await (await fetch(`${baseUrl}/app.js`)).text();

    assert.match(html, /<label[^>]+for="question"/i);
    assert.match(html, /id="status"[^>]+aria-live="polite"/i);
    assert.match(html, /id="baseline-answer"/);
    assert.match(html, /id="grounded-answer"/);
    assert.match(html, /id="evidence-list"/);
    assert.match(html, /example-question/i);
    assert.match(script, /textContent/);
    assert.doesNotMatch(script, /\.innerHTML\s*=/);
  });
});

test('returns 404 for unlisted asset paths', async () => {
  await withServer(async (baseUrl) => {
    const response = await fetch(`${baseUrl}/../../etc/passwd`);
    assert.equal(response.status, 404);
  });
});

test('page exposes the multi-agent review controls and renders results safely', async () => {
  await withServer(async (baseUrl) => {
    const html = await (await fetch(baseUrl)).text();
    const script = await (await fetch(`${baseUrl}/app.js`)).text();

    assert.match(html, /id="agent-form"/);
    assert.match(html, /name="agent-mode"[^>]+value="fact"/);
    assert.match(html, /name="agent-mode"[^>]+value="plan"/);
    assert.match(html, /id="agent-status"[^>]+aria-live="polite"/i);
    assert.match(html, /id="agent-answer"/);
    assert.match(html, /id="agent-candidates"/);
    assert.match(script, /\/api\/agents/);
    assert.doesNotMatch(script, /innerHTML/);
  });
});

test('page lists safety-gate gaps as separate items', async () => {
  await withServer(async (baseUrl) => {
    const html = await (await fetch(baseUrl)).text();
    const script = await (await fetch(`${baseUrl}/app.js`)).text();
    assert.match(html, /id="agent-gaps"/);
    assert.match(script, /agentGaps/);
  });
});

test('page offers example questions for record facts and for what the assistant can do', async () => {
  await withServer(async (baseUrl) => {
    const html = await (await fetch(baseUrl)).text();
    const script = await (await fetch(`${baseUrl}/app.js`)).text();
    assert.match(html, /data-example="How much credit is left\?"/);
    assert.match(html, /data-example="What can you help me with\?"/);
    assert.match(html, /data-example="What is my annual fee\?"/);
    assert.match(script, /'help'/);
  });
});

test('page offers a benefits overview example and a status for it', async () => {
  await withServer(async (baseUrl) => {
    const html = await (await fetch(baseUrl)).text();
    const script = await (await fetch(`${baseUrl}/app.js`)).text();
    assert.match(html, /data-example="What are the benefits of this card\?"/);
    assert.match(script, /'overview'/);
  });
});

test('page warns when the results shown belong to an earlier question', async () => {
  await withServer(async (baseUrl) => {
    const html = await (await fetch(baseUrl)).text();
    const script = await (await fetch(`${baseUrl}/app.js`)).text();
    assert.match(html, /id="stale-note"[^>]+hidden/);
    assert.match(html, /id="stale-note"[^>]+role="status"/);
    assert.match(script, /stale-note/);
    assert.match(html, /previous question/i);
    assert.match(script, /lastComparedQuestion/);
  });
});
