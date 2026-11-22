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

const REMAINING_AMOUNT_QUESTION = /\b(?:how much|what(?:'s| is)?)\b[^?]*\b(?:left|remain(?:s|ing)?|unused)\b|\bremaining (?:credit|balance|benefit|amount)\b|\b(?:credit|benefit)\b[^?]*\b(?:left|remain(?:s|ing)?|unused)\b/i;
const USED_AMOUNT_QUESTION = /\b(?:how much|have i|did i|i['\u2019]ve)\b[^?]*\b(?:used|spent|spend|use)\b/i;
const ANNUAL_FEE_QUESTION = /\bannual fee\b/i;
const ANNIVERSARY_DATE_QUESTION = /\b(?:when|what|which)\b[^?]*\banniversary\b|\banniversary date\b/i;
const HELP_QUESTION = /\b(?:what can you|what do you do|how can you help|help me with|what are you able)\b/i;
// Broad "what are all the benefits" questions have no single best passage, so semantic search cannot answer them;
// the list is built by code from the saved sections, never generated.
const OVERVIEW_QUESTION = /^(?=.*\b(?:benefits|perks|credits)\b)(?:\s*(?:what|which)\s+(?:are\s+)?(?:all\s+)?(?:the\s+|my\s+)?(?:benefits|perks|credits)\b|\s*(?:what|which)\s+(?:benefits|perks|credits)\b|\s*list\b|\s*(?:show|tell)\s+me\s+(?:all\s+)?(?:the\s+|my\s+)?(?:benefits|perks|credits)\b)(?![^?]*\b(?:lounge|lyft|paris|hotel|gas|gasoline|edit|global entry|precheck)\b)/i;

export function isOverviewQuestion(question) {
  return typeof question === 'string' && OVERVIEW_QUESTION.test(question);
}

const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];

const HELP_ANSWER = 'I help you understand the travel benefits on a synthetic household card. I can answer questions about a benefit using a saved public Chase Sapphire Reserve snapshot, with a source and capture date; read amounts such as the credit remaining, the amount used, the annual fee, and the account-anniversary date from the synthetic household record; and, in the multi-agent step, suggest cautious next steps to verify. I cannot see real accounts, make purchases, cancel cards, move money, or give financial advice, and I will say so when the saved sources do not support an answer. Try: "How much credit is left?", "Is the travel credit based on a calendar year?", or "What is my annual fee?"';

function formatAnniversary(value) {
  const match = /^(\d{2})-(\d{2})$/.exec(value ?? '');
  const month = match ? MONTHS[Number(match[1]) - 1] : undefined;
  return month ? `${month} ${Number(match[2])}` : undefined;
}

// Facts that live in the synthetic household record are answered by code, never generated and never cited as public sources.
// The synthetic record only describes the one saved travel credit, so a question that names another benefit must
// never be answered from it (for example "how much of the Lyft credit is left?").
const OTHER_BENEFIT = /\b(?:lyft|edit|lounges?|priority pass|global entry|tsa|precheck|nexus|hotels?|gas|gasoline|dining|groceries|grocery)\b/i;

function answerFromRecord(question, card, remainingAmount) {
  if (OTHER_BENEFIT.test(question)) return null;
  const fromRecord = (text) => ({ answer: `${text} This comes from the synthetic household record, not a public source.`, source: 'household-record' });
  if (REMAINING_AMOUNT_QUESTION.test(question)) {
    return fromRecord(`The synthetic record shows $${remainingAmount} remaining of the $${card.benefitAmount} ${card.benefitName} ($${card.usedAmount} used).`);
  }
  if (USED_AMOUNT_QUESTION.test(question)) {
    return fromRecord(`The synthetic record shows $${card.usedAmount} used of the $${card.benefitAmount} ${card.benefitName}.`);
  }
  if (ANNUAL_FEE_QUESTION.test(question) && Number.isFinite(card.annualFee)) {
    return fromRecord(`The synthetic record lists an annual fee of $${card.annualFee}.`);
  }
  const anniversary = ANNIVERSARY_DATE_QUESTION.test(question) ? formatAnniversary(card.anniversaryDate) : undefined;
  if (anniversary) return fromRecord(`The synthetic record lists the account anniversary as ${anniversary}.`);
  return null;
}

// Routes a question before any retrieval. 'help', 'overview' and 'record' are answered by code with no model call;
// null means the question needs verified retrieval over the public source.
export function classifyQuestion(question, card) {
  if (typeof question !== 'string') return null;
  if (HELP_QUESTION.test(question)) return 'help';
  if (isOverviewQuestion(question)) return 'overview';
  if (!Number.isFinite(card?.benefitAmount) || !Number.isFinite(card?.usedAmount)) return null; // incomplete record: nothing to answer from
  const remaining = calculateRemaining({ benefitAmount: card.benefitAmount, usedAmount: card.usedAmount });
  return answerFromRecord(question, card, remaining) ? 'record' : null;
}

export async function answerQuestion({ question, household, retrievedChunks = [], retrieveEnabled, overview = false, generate = generateWithOllama }) {
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

  if (overview && Array.isArray(retrievedChunks) && retrievedChunks.length > 0) {
    const headings = [...new Set(retrievedChunks.map((chunk) => chunk.heading))];
    const captured = retrievedChunks[0].capturedAt;
    return {
      answer: `The saved ${card.product} snapshot${captured ? ` (captured ${captured})` : ''} lists ${headings.length} benefits: ${headings.join('; ')}. Ask about any one of them for its terms.`,
      remainingAmount,
      citations: retrievedChunks.map(({ id, heading, sourceUrl, capturedAt }) => ({ id, heading, sourceUrl, capturedAt })),
      source: 'benefit-overview',
    };
  }

  const noEvidence = !Array.isArray(retrievedChunks) || retrievedChunks.length === 0;
  if (noEvidence && HELP_QUESTION.test(question)) {
    return { answer: HELP_ANSWER, remainingAmount, citations: [], source: 'app-help' };
  }
  if (noEvidence) {
    const recordAnswer = answerFromRecord(question, card, remainingAmount);
    if (recordAnswer) return { ...recordAnswer, remainingAmount, citations: [] };
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
