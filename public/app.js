const form = document.querySelector('#question-form');
const questionField = document.querySelector('#question');
const questionCount = document.querySelector('#question-count');
const status = document.querySelector('#status');
const submitButton = document.querySelector('#submit-button');
const results = document.querySelector('#results');
const baselineAnswer = document.querySelector('#baseline-answer');
const groundedAnswer = document.querySelector('#grounded-answer');
const benefitAmount = document.querySelector('#benefit-amount');
const usedAmount = document.querySelector('#used-amount');
const remainingAmount = document.querySelector('#remaining-amount');
const evidenceList = document.querySelector('#evidence-list');
const evidenceCount = document.querySelector('#evidence-count');
const currency = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });

function updateCount() {
  questionCount.textContent = `${questionField.value.length} / 300`;
}

function setStatus(message, tone = '') {
  status.textContent = message;
  if (tone) status.dataset.tone = tone;
  else delete status.dataset.tone;
}

function renderEvidence(evidence) {
  evidenceList.replaceChildren();
  evidenceCount.textContent = `${evidence.length} ${evidence.length === 1 ? 'passage' : 'passages'}`;

  if (!evidence.length) {
    const empty = document.createElement('p');
    empty.className = 'empty-state';
    empty.textContent = 'No passage met the relevance threshold. The grounded answer should not guess.';
    evidenceList.append(empty);
    return;
  }

  for (const item of evidence) {
    const card = document.createElement('article');
    card.className = 'evidence-card';

    const header = document.createElement('div');
    header.className = 'evidence-card-top';
    const heading = document.createElement('h3');
    heading.textContent = item.heading || 'Retrieved passage';
    const score = document.createElement('span');
    score.className = 'match-score';
    score.textContent = Number.isFinite(item.score) ? `Match score ${item.score.toFixed(3)}` : 'Retrieved';
    header.append(heading, score);

    const passage = document.createElement('p');
    passage.className = 'evidence-copy';
    passage.textContent = item.text || '';

    const metadata = document.createElement('div');
    metadata.className = 'evidence-meta';
    const source = document.createElement('a');
    try {
      const url = new URL(item.sourceUrl);
      if (url.protocol !== 'https:') throw new Error('Only HTTPS sources are allowed');
      source.href = url.href;
      source.target = '_blank';
      source.rel = 'noopener noreferrer';
      source.textContent = 'View issuer source ↗';
      metadata.append(source);
    } catch {
      const unavailable = document.createElement('span');
      unavailable.textContent = 'Source link unavailable';
      metadata.append(unavailable);
    }
    if (item.capturedAt) {
      const capture = document.createElement('span');
      capture.textContent = `Captured ${item.capturedAt}`;
      metadata.append(capture);
    }

    card.append(header, passage, metadata);
    evidenceList.append(card);
  }
}

function renderComparison(data) {
  baselineAnswer.textContent = data.baseline;
  groundedAnswer.textContent = data.grounded;
  benefitAmount.textContent = currency.format(data.benefitAmount);
  usedAmount.textContent = currency.format(data.usedAmount);
  remainingAmount.textContent = currency.format(data.remainingAmount);
  renderEvidence(Array.isArray(data.evidence) ? data.evidence : []);
}

questionField.addEventListener('input', updateCount);
for (const button of document.querySelectorAll('[data-example]')) {
  button.addEventListener('click', () => {
    questionField.value = button.dataset.example;
    updateCount();
    questionField.focus();
  });
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!form.reportValidity()) return;

  submitButton.disabled = true;
  results.setAttribute('aria-busy', 'true');
  setStatus('Retrieving public benefit information and comparing answers…');

  try {
    const response = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ question: questionField.value.trim() }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);

    renderComparison(data);
    setStatus(data.evidence?.length ? 'Comparison complete. Retrieved evidence is shown below.' : 'Comparison complete. No source passed the relevance threshold.', data.evidence?.length ? 'success' : '');
  } catch (error) {
    setStatus(error.message || 'The local assistant could not complete the request.', 'error');
  } finally {
    submitButton.disabled = false;
    results.setAttribute('aria-busy', 'false');
  }
});

updateCount();
