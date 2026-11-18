import { validateOllamaBaseUrl } from './local-ollama.mjs';

const OLLAMA_BASE_URL = process.env.OLLAMA_HOST ?? 'http://localhost:11434';
const CHAT_MODEL = 'llama3.2';

export function calculateRemaining({ benefitAmount, usedAmount }) {
  if (!Number.isFinite(benefitAmount) || !Number.isFinite(usedAmount) || benefitAmount < 0 || usedAmount < 0) {
    throw new Error('Benefit and usage amounts must be non-negative numbers');
  }
  if (usedAmount > benefitAmount) throw new Error('Usage cannot exceed the benefit amount');
  return benefitAmount - usedAmount;
}

export async function generateWithOllama(prompt, { baseUrl = OLLAMA_BASE_URL, model = CHAT_MODEL, fetchImpl = fetch } = {}) {
  const endpoint = `${validateOllamaBaseUrl(baseUrl)}/api/chat`;
  let response;
  try {
    response = await fetchImpl(endpoint, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model,
        messages: [
          {
            role: 'system',
            content: 'Answer the user using only the supplied retrieved issuer passages. Do not use prior knowledge or infer missing terms. If the passages do not support an answer, say the evidence is insufficient. Do not calculate numeric values or invent citations; application code displays calculations and source citations separately.',
          },
          { role: 'user', content: prompt },
        ],
        stream: false,
        options: { temperature: 0 },
      }),
    });
  } catch (error) {
    throw new Error(`Ollama answer request failed (model "${model}", endpoint ${endpoint}). Is Ollama running and the model installed? ${error.message}`);
  }
  if (!response.ok) {
    const detail = await response.text().catch(() => '');
    throw new Error(`Ollama answer request failed with HTTP ${response.status}${detail ? `: ${detail}` : ''}`);
  }
  const data = await response.json().catch(() => null);
  const answer = data?.message?.content;
  if (typeof answer !== 'string' || !answer.trim()) throw new Error('Ollama returned an empty answer');
  return answer.trim();
}

export async function answerQuestion({ question, household, retrievedChunks = [], retrieveEnabled, generate = generateWithOllama }) {
  if (typeof question !== 'string' || !question.trim()) throw new Error('Question must be non-empty');
  const card = household?.card;
  if (!card) throw new Error('Synthetic household card is required');
  const remainingAmount = calculateRemaining({ benefitAmount: card.benefitAmount, usedAmount: card.usedAmount });

  if (!retrieveEnabled) {
    return {
      answer: `The synthetic record shows $${remainingAmount} remaining, but I cannot determine the benefit's reset rule from that record alone.`,
      remainingAmount,
      citations: [],
    };
  }

  if (!Array.isArray(retrievedChunks) || retrievedChunks.length === 0) {
    return {
      answer: 'I have insufficient evidence in the retrieved public sources to answer that question.',
      remainingAmount,
      citations: [],
    };
  }

  const evidence = retrievedChunks.map((chunk, index) => `[S${index + 1}] ${chunk.heading}\n${chunk.text}`).join('\n\n');
  const prompt = [
    `Question: ${question}`,
    `Synthetic benefit record: ${card.benefitName}, limit $${card.benefitAmount}, used $${card.usedAmount}, remaining $${remainingAmount}, account anniversary ${card.anniversaryDate}.`,
    'For this cycle comparison, if evidence says each account anniversary year, answer that cycle directly. Do not infer a specific date not stated in the passage.',
    'Retrieved public-source evidence (answer only when supported by these passages):',
    evidence,
  ].join('\n\n');
  const answer = await generate(prompt);
  if (typeof answer !== 'string' || !answer.trim()) throw new Error('Answer generator returned an empty response');

  return {
    answer: answer.trim(),
    remainingAmount,
    citations: retrievedChunks.map(({ id, heading, sourceUrl, capturedAt }) => ({ id, heading, sourceUrl, capturedAt })),
  };
}
