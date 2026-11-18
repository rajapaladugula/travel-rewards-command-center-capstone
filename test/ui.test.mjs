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
