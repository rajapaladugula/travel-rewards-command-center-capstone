import { validateOllamaBaseUrl } from './local-ollama.mjs';

const DEFAULT_BASE_URL = process.env.OLLAMA_HOST ?? 'http://localhost:11434';
const DEFAULT_MODEL = 'nomic-embed-text';

export async function embedTexts(texts, { baseUrl = DEFAULT_BASE_URL, model = DEFAULT_MODEL, fetchImpl = fetch } = {}) {
  if (!Array.isArray(texts) || texts.length === 0 || texts.some((text) => typeof text !== 'string' || !text.trim())) {
    throw new Error('Embedding input must be a non-empty array of non-empty strings');
  }

  const endpoint = `${validateOllamaBaseUrl(baseUrl)}/api/embed`;
  let response;
  try {
    response = await fetchImpl(endpoint, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model, input: texts }),
    });
  } catch (error) {
    throw new Error(`Ollama embedding request failed (model "${model}", endpoint ${endpoint}). Is Ollama running and the model installed? ${error.message}`);
  }

  if (!response.ok) {
    const detail = await response.text().catch(() => '');
    throw new Error(`Ollama embedding request failed with HTTP ${response.status}${detail ? `: ${detail}` : ''}`);
  }

  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error('Ollama returned an invalid embeddings response');
  }

  const embeddings = data?.embeddings;
  const valid = Array.isArray(embeddings)
    && embeddings.length === texts.length
    && embeddings.every((vector) => Array.isArray(vector) && vector.length > 0 && vector.every(Number.isFinite))
    && embeddings.every((vector) => vector.length === embeddings[0].length);
  if (!valid) throw new Error(`Ollama returned an invalid embeddings response; expected ${texts.length} equal-length numeric vectors`);
  return embeddings;
}
