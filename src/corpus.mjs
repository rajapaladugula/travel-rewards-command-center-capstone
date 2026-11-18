import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const SOURCE_PATH = resolve(ROOT, 'data/chase-sapphire-reserve.md');
const HOUSEHOLD_PATH = resolve(ROOT, 'data/synthetic-household.json');
const MAX_ESTIMATED_TOKENS = 450;

function estimateTokens(text) {
  return Math.ceil(text.length / 4);
}

function parseFrontmatter(markdown) {
  const match = markdown.match(/^---\s*\n([\s\S]*?)\n---\s*\n?/);
  if (!match) throw new Error('Source is missing frontmatter metadata');
  const metadata = {};
  for (const line of match[1].split('\n')) {
    const separator = line.indexOf(':');
    if (separator < 1) throw new Error(`Invalid source metadata line: ${line}`);
    metadata[line.slice(0, separator).trim()] = line.slice(separator + 1).trim();
  }
  return { metadata, body: markdown.slice(match[0].length) };
}

function splitLongSection(heading, text) {
  const sentences = text.match(/[^.!?]+[.!?]+|[^.!?]+$/g)?.map((sentence) => sentence.trim()).filter(Boolean) ?? [];
  if (!sentences.length) return [];

  const pieces = [];
  let current = [];
  for (const sentence of sentences) {
    let candidate = [...current, sentence];
    if (current.length && estimateTokens(candidate.join(' ')) > MAX_ESTIMATED_TOKENS) {
      pieces.push(current.join(' '));
      const overlap = current.at(-1);
      current = [overlap, sentence];
      candidate = current;
    }
    if (estimateTokens(candidate.join(' ')) > MAX_ESTIMATED_TOKENS) {
      const words = sentence.split(/\s+/);
      current = [];
      for (let start = 0; start < words.length; start += MAX_ESTIMATED_TOKENS * 4) {
        pieces.push(words.slice(start, start + MAX_ESTIMATED_TOKENS * 4).join(' '));
      }
    } else {
      current = candidate;
    }
  }
  if (current.length) pieces.push(current.join(' '));
  return pieces.map((piece) => ({ heading, text: piece }));
}

export function parseBenefitMarkdown(markdown, metadata) {
  if (typeof markdown !== 'string' || !markdown.trim()) throw new Error('Source must contain at least one benefit section');
  const { product, issuer, sourceUrl, capturedAt } = metadata ?? {};
  if (!product || !issuer) throw new Error('Source product and issuer metadata are required');
  if (!/^https?:\/\//i.test(sourceUrl ?? '')) throw new Error('Source URL must be an HTTP(S) URL');
  if (!/^\d{4}-\d{2}-\d{2}$/.test(capturedAt ?? '')) throw new Error('Source capture date must use YYYY-MM-DD format');

  const sections = [...markdown.matchAll(/^###\s+(.+)\s*$/gm)];
  if (!sections.length) throw new Error('Source must contain at least one benefit section');

  return sections.flatMap((section, index) => {
    const start = section.index + section[0].length;
    const end = sections[index + 1]?.index ?? markdown.length;
    const text = markdown.slice(start, end).trim();
    if (!text) return [];
    return splitLongSection(section[1].trim(), text).map((piece, pieceIndex) => ({
      id: `${slug(section[1])}-${pieceIndex + 1}`,
      product,
      issuer,
      heading: piece.heading,
      text: piece.text,
      sourceUrl,
      capturedAt,
    }));
  });
}

function slug(value) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}

export async function loadCorpus() {
  const markdown = await readFile(SOURCE_PATH, 'utf8');
  const { metadata, body } = parseFrontmatter(markdown);
  const chunks = parseBenefitMarkdown(body, metadata);
  if (!chunks.length) throw new Error('Public source contains no non-empty benefit sections');
  return chunks;
}

export async function loadSyntheticHousehold() {
  const data = JSON.parse(await readFile(HOUSEHOLD_PATH, 'utf8'));
  const card = data.card;
  if (!card || !Number.isFinite(card.benefitAmount) || !Number.isFinite(card.usedAmount)) {
    throw new Error('Synthetic card benefit amounts must be numeric');
  }
  return data;
}
