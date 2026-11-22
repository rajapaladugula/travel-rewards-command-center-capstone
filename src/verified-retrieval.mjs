import { retrieve } from './retriever.mjs';

// Retrieve loosely, verify tightly: a lower similarity floor finds rephrased questions, and a strict yes/no judge
// (plus one query rewrite) decides whether a passage really states the asked fact. Anything unverified is dropped,
// so the failure mode stays "refuse", never "answer from a related-looking passage".
export const LOOSE_MIN_SCORE = 0.5;
export const LOOSE_TOP_K = 3;
const MAX_REWRITE_LENGTH = 200;

export function parseVerdict(text) {
  const match = /^\W*(yes|no)\b/i.exec(typeof text === 'string' ? text : '');
  return match ? match[1].toLowerCase() : 'unknown';
}

export function sanitizeRewrite(text, original) {
  if (typeof text !== 'string') return null;
  const line = text.split('\n').map((part) => part.trim()).find(Boolean);
  if (!line) return null;
  const cleaned = line.replace(/^["'`]+|["'`]+$/g, '').trim().slice(0, MAX_REWRITE_LENGTH);
  if (!cleaned || cleaned.toLowerCase() === String(original ?? '').trim().toLowerCase()) return null;
  return cleaned;
}

const STOPWORDS = new Set((
  'a an the is are was were be been do does did can could would should will may might i me my we you your it its this that these those '
  + 'of for to in on at by with from as and or but not no yes if so any all some there here what which who whom when where why how '
  + 'have has had get gets give gives include includes offer offers provide provides cover covers card cards benefit benefits perk perks '
  + 'much many more most free also about into than then'
).split(' '));
const MAX_DISTINCTIVE_DOCUMENT_FREQUENCY = 2;

const stem = (word) => (word.length > 3 && word.endsWith('s') && !word.endsWith('ss') ? word.slice(0, -1) : word);

export function contentTokens(text) {
  const tokens = String(text ?? '').toLowerCase().replace(/[^a-z0-9$]+/g, ' ').split(' ').filter(Boolean);
  return [...new Set(tokens.filter((word) => !STOPWORDS.has(word)).map(stem))];
}

// A passage can only be verified for a question if they share a word that is distinctive in this source
// (appears in at most two of the card's passages). This is deterministic, so a weak judge cannot say YES to a
// passage that has nothing in common with the question.
export function hasDistinctiveOverlap(question, chunk, allChunks) {
  const documentFrequency = new Map();
  for (const other of allChunks) {
    for (const token of new Set(contentTokens(`${other.heading} ${other.text}`))) {
      documentFrequency.set(token, (documentFrequency.get(token) ?? 0) + 1);
    }
  }
  const passageTokens = new Set(contentTokens(`${chunk.heading} ${chunk.text}`));
  return contentTokens(question).some((token) => passageTokens.has(token) && (documentFrequency.get(token) ?? 0) <= MAX_DISTINCTIVE_DOCUMENT_FREQUENCY);
}

async function verify({ question, candidates, judge, chunks }) {
  const kept = [];
  for (const candidate of candidates) {
    if (!hasDistinctiveOverlap(question, candidate, chunks)) continue;
    if (parseVerdict(await judge(question, candidate)) === 'yes') kept.push(candidate);
  }
  return kept;
}

export async function verifiedRetrieve({
  question,
  chunks,
  card,
  retrieveFn = retrieve,
  judge,
  rewrite,
  minScore = 0.7,
  looseMinScore = LOOSE_MIN_SCORE,
  useRewrite = true,
  verifyStrict = true,
}) {
  const search = (query, floor, topK) => retrieveFn({ query, chunks, product: card.product, issuer: card.issuer, minScore: floor, ...(topK ? { topK } : {}) });

  const strict = await search(question, minScore);
  let judged = 0;
  if (strict.length > 0) {
    if (!verifyStrict) return { results: strict, path: 'strict', judged: 0, rewrites: 0 };
    judged += strict.length;
    const confirmed = await verify({ question, candidates: strict, judge, chunks });
    if (confirmed.length > 0) return { results: confirmed, path: 'strict', judged, rewrites: 0 };
  }

  const firstPass = await search(question, looseMinScore, LOOSE_TOP_K);
  judged += firstPass.length;
  const verified = await verify({ question, candidates: firstPass, judge, chunks });
  if (verified.length > 0) return { results: verified, path: 'verified', judged, rewrites: 0 };

  if (useRewrite && typeof rewrite === 'function') {
    const rewritten = sanitizeRewrite(await rewrite(question), question);
    if (rewritten) {
      const secondPass = await search(rewritten, looseMinScore, LOOSE_TOP_K);
      judged += secondPass.length;
      // Always judged against the user's original question, never the rewrite.
      const confirmed = await verify({ question, candidates: secondPass, judge, chunks });
      if (confirmed.length > 0) return { results: confirmed, path: 'rewritten', judged, rewrites: 1 };
      return { results: [], path: 'none', judged, rewrites: 1 };
    }
  }
  return { results: [], path: 'none', judged, rewrites: 0 };
}

// Comparative verification: a small model chooses between candidates more reliably than it judges one in isolation.
export function parseSelection(text, count) {
  const match = /^\W*(?:passage\s*)?(\d+)\W*$/i.exec(typeof text === 'string' ? text.trim() : '');
  if (!match) return null;
  const index = Number(match[1]);
  return Number.isInteger(index) && index >= 1 && index <= count ? index : null;
}

export async function selectVerified({
  question,
  chunks,
  card,
  retrieveFn = retrieve,
  select,
  rewrite,
  minScore = 0.7,
  looseMinScore = LOOSE_MIN_SCORE,
  useRewrite = true,
}) {
  const search = (query, floor, topK) => retrieveFn({ query, chunks, product: card.product, issuer: card.issuer, minScore: floor, ...(topK ? { topK } : {}) });
  const choose = async (candidates) => {
    const unique = [...new Map(candidates.map((candidate) => [candidate.id, candidate])).values()]
      .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
      .filter((candidate) => hasDistinctiveOverlap(question, candidate, chunks))
      .slice(0, LOOSE_TOP_K);
    if (unique.length === 0) return { picked: null, asked: 0 };
    const index = parseSelection(await select(question, unique), unique.length);
    return { picked: index ? unique[index - 1] : null, asked: 1 };
  };

  const first = await choose([...(await search(question, minScore)), ...(await search(question, looseMinScore, LOOSE_TOP_K))]);
  if (first.picked) return { results: [first.picked], path: 'selected', judged: first.asked, rewrites: 0 };

  if (useRewrite && typeof rewrite === 'function') {
    const rewritten = sanitizeRewrite(await rewrite(question), question);
    if (rewritten) {
      const second = await choose(await search(rewritten, looseMinScore, LOOSE_TOP_K));
      if (second.picked) return { results: [second.picked], path: 'rewritten', judged: first.asked + second.asked, rewrites: 1 };
      return { results: [], path: 'none', judged: first.asked + second.asked, rewrites: 1 };
    }
  }
  return { results: [], path: 'none', judged: first.asked, rewrites: 0 };
}
