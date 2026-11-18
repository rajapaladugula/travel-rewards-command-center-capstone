import test from 'node:test';
import assert from 'node:assert/strict';
import { loadCorpus, loadSyntheticHousehold, parseBenefitMarkdown } from '../src/corpus.mjs';

const SOURCE_URL = 'https://creditcards.chase.com/rewards-credit-cards/sapphire/reserve';
const CAPTURED_AT = '2026-10-27';

test('parses benefit sections and preserves source metadata', () => {
  const chunks = parseBenefitMarkdown(
    '### $300 annual travel credit\n\nUp to $300 in statement credits for travel purchases each account anniversary year.\n\n### Lounge access\n\nVisit airport lounges.',
    { product: 'Chase Sapphire Reserve', issuer: 'Chase', sourceUrl: SOURCE_URL, capturedAt: CAPTURED_AT },
  );

  assert.equal(chunks.length, 2);
  assert.equal(chunks[0].heading, '$300 annual travel credit');
  assert.match(chunks[0].text, /account anniversary year/);
  assert.equal(chunks[0].product, 'Chase Sapphire Reserve');
  assert.equal(chunks[0].issuer, 'Chase');
  assert.equal(chunks[0].sourceUrl, SOURCE_URL);
  assert.equal(chunks[0].capturedAt, CAPTURED_AT);
});

test('splits long benefit sections at sentence boundaries with overlapping context', () => {
  const sentences = Array.from({ length: 90 }, (_, i) => `Sentence ${i + 1} explains a distinct card benefit condition.`);
  const chunks = parseBenefitMarkdown(
    `### Long benefit\n\n${sentences.join(' ')}`,
    { product: 'Chase Sapphire Reserve', issuer: 'Chase', sourceUrl: SOURCE_URL, capturedAt: CAPTURED_AT },
  );

  assert.ok(chunks.length > 1);
  assert.ok(chunks.every((chunk) => Math.ceil(chunk.text.length / 4) <= 450));
  assert.ok(chunks[1].text.startsWith('Sentence '));
  assert.ok(chunks[0].text.split('. ').at(-1).slice(0, 12).includes('Sentence'));
  assert.ok(chunks[1].text.includes(chunks[0].text.split('. ').at(-1)));
});

test('rejects empty source and missing source metadata', () => {
  assert.throws(
    () => parseBenefitMarkdown('', { product: 'Chase Sapphire Reserve', issuer: 'Chase', sourceUrl: SOURCE_URL, capturedAt: CAPTURED_AT }),
    /at least one benefit section/i,
  );
  assert.throws(
    () => parseBenefitMarkdown('### Travel credit\n\nText.', { product: 'Chase Sapphire Reserve', issuer: 'Chase', sourceUrl: '', capturedAt: CAPTURED_AT }),
    /source url/i,
  );
});

test('loads the public corpus and synthetic card fixture', async () => {
  const chunks = await loadCorpus();
  const household = await loadSyntheticHousehold();

  assert.ok(chunks.some((chunk) => /account anniversary year/i.test(chunk.text)));
  assert.ok(chunks.every((chunk) => chunk.sourceUrl === SOURCE_URL && chunk.capturedAt === CAPTURED_AT));
  assert.equal(household.card.benefitAmount, 300);
  assert.equal(household.card.usedAmount, 180);
  assert.equal(household.card.anniversaryDate, '08-15');
});
