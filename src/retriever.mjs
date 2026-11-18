import { embedTexts } from './embeddings.mjs';

function cosineSimilarity(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length || !left.length) {
    throw new Error('Embedding dimensions must match and be non-empty');
  }
  let dot = 0;
  let leftMagnitude = 0;
  let rightMagnitude = 0;
  for (let index = 0; index < left.length; index += 1) {
    const a = left[index];
    const b = right[index];
    if (!Number.isFinite(a) || !Number.isFinite(b)) throw new Error('Embedding values must be finite numbers');
    dot += a * b;
    leftMagnitude += a * a;
    rightMagnitude += b * b;
  }
  if (leftMagnitude === 0 || rightMagnitude === 0) return 0;
  return dot / (Math.sqrt(leftMagnitude) * Math.sqrt(rightMagnitude));
}

export async function retrieve({
  query,
  chunks,
  product,
  issuer,
  topK = 3,
  minScore = 0.7,
  embed = embedTexts,
}) {
  if (typeof query !== 'string' || !query.trim()) throw new Error('Retrieval query must be non-empty');
  if (!Number.isInteger(topK) || topK < 1) throw new Error('topK must be a positive integer');
  if (!Number.isFinite(minScore) || minScore < -1 || minScore > 1) throw new Error('minScore must be between -1 and 1');

  const eligible = (chunks ?? []).filter((chunk) => chunk.product === product && chunk.issuer === issuer);
  if (!eligible.length) return [];

  const vectors = await embed([query, ...eligible.map((chunk) => chunk.text)]);
  if (!Array.isArray(vectors) || vectors.length !== eligible.length + 1) {
    throw new Error('Embedding count must match the query and eligible passages');
  }
  const [queryVector, ...documentVectors] = vectors;

  return eligible
    .map((chunk, index) => ({ ...chunk, score: cosineSimilarity(queryVector, documentVectors[index]) }))
    .filter((chunk) => chunk.score >= minScore)
    .sort((left, right) => right.score - left.score)
    .slice(0, topK);
}
