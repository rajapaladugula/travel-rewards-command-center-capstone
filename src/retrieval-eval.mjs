import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { loadCorpus, loadSyntheticHousehold } from './corpus.mjs';
import { retrieve } from './retriever.mjs';
import { verifiedRetrieve, selectVerified } from './verified-retrieval.mjs';
import { judgeWithOllama, rewriteWithOllama, selectWithOllama } from './local-judge.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const CASES_PATH = resolve(ROOT, 'evaluation/retrieval-cases.json');
const RESULTS_PATH = resolve(ROOT, 'evaluation/retrieval-results.json');
const DEFAULT_MIN_SCORE = 0.7;

// Scores retrieval alone (no answer generation): did the top passage match the labeled section, and did unsupported
// questions get no passage? Reports contain case IDs and aggregates only, never question text.
export async function evaluateRetrieval({ cases, chunks, card, minScore = DEFAULT_MIN_SCORE, retrieveFn = retrieve }) {
  const answerable = { total: 0, correct: 0, wrongPassage: 0, abstained: 0 };
  const unsupported = { total: 0, abstained: 0, falseAnswers: 0 };
  const results = [];

  for (const { id, question, expectedHeading } of cases) {
    const found = await retrieveFn({ query: question, chunks, product: card.product, issuer: card.issuer, minScore });
    const top = found[0];
    let outcome;
    if (expectedHeading) {
      answerable.total += 1;
      if (!top) { answerable.abstained += 1; outcome = 'abstained'; }
      else if (top.heading === expectedHeading) { answerable.correct += 1; outcome = 'correct'; }
      else { answerable.wrongPassage += 1; outcome = 'wrong-passage'; }
    } else {
      unsupported.total += 1;
      if (!top) { unsupported.abstained += 1; outcome = 'correct-abstention'; }
      else { unsupported.falseAnswers += 1; outcome = 'false-answer'; }
    }
    results.push({ id, outcome });
  }

  return {
    minScore,
    answerable,
    unsupported,
    answerableRecall: answerable.total ? answerable.correct / answerable.total : null,
    cases: results,
  };
}

const SPLITS = ['development', 'holdout', 'holdout2', 'holdout3', 'holdout4'];

// Runs each retrieval variant on each split. The holdout split is report-only: nothing here tunes a threshold or prompt.
export async function compareVariants({ cases, chunks, card, retrieveFn = retrieve, judge, rewrite, select, splits = SPLITS }) {
  const verdicts = new Map();
  const cachedJudge = async (question, chunk) => {
    const key = `${question}\u0000${chunk.id}`;
    if (!verdicts.has(key)) verdicts.set(key, await judge(question, chunk));
    return verdicts.get(key);
  };
  const variants = {
    strict: (question) => retrieveFn({ query: question, chunks, product: card.product, issuer: card.issuer, minScore: DEFAULT_MIN_SCORE }),
    'verified-loose': async (question) => (await verifiedRetrieve({ question, chunks, card, retrieveFn, judge: cachedJudge, rewrite, useRewrite: true, verifyStrict: false })).results,
    'verified-all': async (question) => (await verifiedRetrieve({ question, chunks, card, retrieveFn, judge: cachedJudge, rewrite, useRewrite: true, verifyStrict: true })).results,
    ...(select ? { 'select-all': async (question) => (await selectVerified({ question, chunks, card, retrieveFn, select, rewrite, useRewrite: true })).results } : {}),
  };

  const report = { minScore: DEFAULT_MIN_SCORE, variants: {} };
  for (const [name, run] of Object.entries(variants)) {
    report.variants[name] = {};
    for (const split of splits) {
      const splitCases = cases.filter((c) => c.split === split);
      report.variants[name][split] = await evaluateRetrieval({
        cases: splitCases,
        chunks,
        card,
        minScore: DEFAULT_MIN_SCORE,
        retrieveFn: ({ query }) => run(query),
      });
    }
  }
  return report;
}

async function main() {
  const splitFlag = process.argv.indexOf('--splits');
  const splits = splitFlag > -1 ? process.argv[splitFlag + 1].split(',') : SPLITS;
  const { cases } = JSON.parse(await readFile(CASES_PATH, 'utf8'));
  const [chunks, household] = await Promise.all([loadCorpus(), loadSyntheticHousehold()]);
  const report = await compareVariants({ cases, chunks, card: household.card, judge: judgeWithOllama, rewrite: rewriteWithOllama, select: selectWithOllama, splits });
  // A partial run (for example development only, while tuning) never overwrites the committed full report.
  if (splits.length === SPLITS.length) await writeFile(RESULTS_PATH, `${JSON.stringify(report, null, 2)}\n`);
  for (const [name, bySplit] of Object.entries(report.variants)) {
    for (const split of splits) {
      const { answerable: a, unsupported: u } = bySplit[split];
      console.log(`${name.padEnd(16)} ${split.padEnd(11)} answerable ${a.correct}/${a.total} correct (${a.wrongPassage} wrong, ${a.abstained} abstained) | unsupported ${u.abstained}/${u.total} refused (${u.falseAnswers} false answers)`);
    }
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => { console.error(`Retrieval evaluation failed: ${error.message}`); process.exitCode = 1; });
}
