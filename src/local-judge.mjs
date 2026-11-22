import { validateOllamaBaseUrl } from './local-ollama.mjs';

const OLLAMA_BASE_URL = process.env.OLLAMA_HOST ?? 'http://localhost:11434';
const CHAT_MODEL = 'llama3.2';

const JUDGE_SYSTEM = [
  'You check whether one passage from a card issuer\'s benefits page answers a question.',
  'Answer YES only if the passage itself states the specific fact the question asks about.',
  'Answer NO if the passage is merely related, shares a word with the question, or does not state the asked fact.',
  'Examples: Question "Is there a baggage fee waiver?" with a passage about airport lounge access is NO. Question "How many guests can enter the lounge?" with a passage saying lounge access includes up to two guests is YES.',
  'Reply with exactly one word: YES or NO.',
].join(' ');

const REWRITE_SYSTEM = [
  'Rewrite the user\'s question as one short search query (at most 15 words) using words a card issuer\'s benefits page would likely use.',
  'Do not answer the question, add facts, or add numbers. Output only the query.',
].join(' ');

async function chat({ system, user, maxTokens, baseUrl = OLLAMA_BASE_URL, model = CHAT_MODEL, fetchImpl = fetch }) {
  const endpoint = `${validateOllamaBaseUrl(baseUrl)}/api/chat`;
  let response;
  try {
    response = await fetchImpl(endpoint, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model,
        messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
        stream: false,
        options: { temperature: 0, num_predict: maxTokens },
      }),
    });
  } catch (error) {
    throw new Error(`Ollama verification request failed (model "${model}", endpoint ${endpoint}). Is Ollama running? ${error.message}`);
  }
  if (!response.ok) throw new Error(`Ollama verification request failed with HTTP ${response.status}`);
  const data = await response.json().catch(() => null);
  const text = data?.message?.content;
  if (typeof text !== 'string') throw new Error('Ollama returned an empty verification response');
  return text.trim();
}

export function judgeWithOllama(question, chunk, options = {}) {
  return chat({ system: JUDGE_SYSTEM, user: `Question: ${question}\n\nPassage (${chunk.heading}): ${chunk.text}`, maxTokens: 4, ...options });
}

export function rewriteWithOllama(question, options = {}) {
  return chat({ system: REWRITE_SYSTEM, user: question, maxTokens: 40, ...options });
}

const SELECT_SYSTEM = [
  'You choose which numbered passage from a card issuer\'s benefits page states the specific fact a question asks about.',
  'Pick a passage only if it itself states that fact. A passage that is merely related, or only shares a word with the question, does not count.',
  'If no passage states the fact, answer NONE.',
  'Example: Question "Is there a baggage fee waiver?" with passage 1 about airport lounge access and passage 2 about ride credits is NONE.',
  'Reply with only the passage number or the word NONE.',
].join(' ');

export function selectWithOllama(question, candidates, options = {}) {
  const passages = candidates.map((candidate, index) => `Passage ${index + 1} (${candidate.heading}): ${candidate.text}`).join('\n\n');
  return chat({ system: SELECT_SYSTEM, user: `Question: ${question}\n\n${passages}`, maxTokens: 6, ...options });
}
