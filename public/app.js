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

const staleNote = document.querySelector('#stale-note');
let lastComparedQuestion = null;

function updateCount() {
  questionCount.textContent = `${questionField.value.length} / 300`;
}

function setStatus(message, tone = '') {
  status.textContent = message;
  if (tone) status.dataset.tone = tone;
  else delete status.dataset.tone;
}

function renderEvidence(evidence, recordAnswer = '') {
  evidenceList.replaceChildren();
  evidenceCount.textContent = `${evidence.length} ${evidence.length === 1 ? 'passage' : 'passages'}`;

  if (!evidence.length) {
    const empty = document.createElement('p');
    empty.className = 'empty-state';
    empty.textContent = recordAnswer === 'help' ? 'No source was needed: this explains what the assistant can do.' : recordAnswer ? 'No public passage was needed: this comes from the synthetic household record.' : 'No passage met the relevance threshold. The grounded answer should not guess.';
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
  renderEvidence(Array.isArray(data.evidence) ? data.evidence : [], ['record-answer', 'help'].includes(data.status) ? (data.status === 'help' ? 'help' : 'record') : '');
}

questionField.addEventListener('input', () => {
  updateCount();
  staleNote.hidden = lastComparedQuestion === null || questionField.value.trim() === lastComparedQuestion;
});
for (const button of document.querySelectorAll('[data-example]')) {
  button.addEventListener('click', () => {
    questionField.value = button.dataset.example;
    updateCount();
    staleNote.hidden = lastComparedQuestion === null || questionField.value.trim() === lastComparedQuestion;
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
    lastComparedQuestion = questionField.value.trim();
    staleNote.hidden = true;
    if (data.status === 'overview') setStatus('Comparison complete. This lists every section of the saved source; ask about one for its terms.', 'success');
    else if (data.status === 'help') setStatus('Comparison complete. This describes what the assistant can and cannot do; no source was needed.', 'success');
    else if (data.status === 'record-answer') setStatus('Comparison complete. This answer comes from the synthetic household record, so no public passage was needed.', 'success');
    else setStatus(data.evidence?.length ? 'Comparison complete. Retrieved evidence is shown below.' : 'Comparison complete. No source passed the relevance threshold.', data.evidence?.length ? 'success' : '');
  } catch (error) {
    setStatus(error.message || 'The local assistant could not complete the request.', 'error');
  } finally {
    submitButton.disabled = false;
    results.setAttribute('aria-busy', 'false');
  }
});

const agentForm = document.querySelector('#agent-form');
const agentSection = document.querySelector('#agent-section');
const agentSubmit = document.querySelector('#agent-submit');
const agentStatus = document.querySelector('#agent-status');
const agentResult = document.querySelector('#agent-result');
const agentOutcome = document.querySelector('#agent-outcome');
const agentMeta = document.querySelector('#agent-meta');
const agentAnswer = document.querySelector('#agent-answer');
const agentCandidates = document.querySelector('#agent-candidates');
const agentGaps = document.querySelector('#agent-gaps');
const agentCitations = document.querySelector('#agent-citations');
const OUTCOME_LABELS = {
  answered: 'ANSWERED',
  abstained: 'ABSTAINED',
  clarification_required: 'NEEDS CLARIFICATION',
  failed: 'SAFELY STOPPED',
};

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderCandidates(data) {
  agentCandidates.replaceChildren();
  const reviews = new Map((data.candidate_reviews || []).map((review) => [review.candidate_id, review]));
  for (const candidate of data.candidates || []) {
    const review = reviews.get(candidate.candidate_id);
    const card = element('article', 'agent-candidate');
    card.dataset.selected = String(candidate.candidate_id === data.selected_candidate_id);
    card.append(element('h4', '', candidate.candidate_id === data.selected_candidate_id ? `${candidate.candidate_id} (selected)` : candidate.candidate_id));
    const steps = element('ol');
    for (const step of candidate.steps || []) {
      const refs = (step.evidence_ids || []).length ? ` [${step.evidence_ids.join(', ')}]` : '';
      steps.append(element('li', '', `${step.kind}: ${step.description}${refs}`));
    }
    card.append(steps);
    if (review?.scores) {
      const { grounding, goal_fit, feasibility, clarity } = review.scores;
      card.append(element('p', 'scores', `Critic scores — grounding ${grounding}/2, goal fit ${goal_fit}/2, feasibility ${feasibility}/2, clarity ${clarity}/2 (total ${grounding + goal_fit + feasibility + clarity}/8)`));
    }
    for (const failure of review?.hard_failures || []) card.append(element('p', 'failures', `Rejected: ${failure}`));
    agentCandidates.append(card);
  }
  if (data.mode === 'plan' && !data.selected_candidate_id) {
    agentCandidates.append(element('p', 'agent-review-note', 'No plan was recommended. Human review is required before taking any step; this prototype never acts on your behalf.'));
  }
}

function renderAgentCitations(citations) {
  agentCitations.replaceChildren();
  for (const item of citations || []) {
    const card = element('article', 'evidence-card');
    const header = element('div', 'evidence-card-top');
    header.append(element('h3', '', item.heading || 'Cited passage'), element('span', 'match-score', item.id || ''));
    card.append(header, element('p', 'evidence-copy', item.excerpt || ''));
    const metadata = element('div', 'evidence-meta');
    try {
      const url = new URL(item.source_url);
      if (url.protocol !== 'https:') throw new Error('Only HTTPS sources are allowed');
      const link = element('a', '', 'View issuer source ↗');
      link.href = url.href;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      metadata.append(link);
    } catch {
      metadata.append(element('span', '', 'Source link unavailable'));
    }
    if (item.captured_at) metadata.append(element('span', '', `Captured ${item.captured_at}`));
    card.append(metadata);
    agentCitations.append(card);
  }
}

function renderAgentResult(data) {
  agentResult.hidden = false;
  agentOutcome.textContent = OUTCOME_LABELS[data.status] || data.status;
  agentMeta.textContent = `${data.mode} mode${data.retry_count ? ' · 1 replan' : ''}`;
  const gapsMarker = data.answer.indexOf('gaps: ');
  const gaps = gapsMarker >= 0 ? data.answer.slice(gapsMarker + 6).split('; ').filter(Boolean) : [];
  agentAnswer.textContent = gaps.length ? data.answer.slice(0, gapsMarker + 5) : data.answer;
  agentGaps.replaceChildren(...gaps.map((gap) => element('li', '', gap)));
  agentGaps.hidden = !gaps.length;
  renderCandidates(data);
  renderAgentCitations(data.evidence?.citations);
}

agentForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!form.reportValidity()) return;
  const mode = agentForm.querySelector('input[name="agent-mode"]:checked').value;

  agentSubmit.disabled = true;
  agentSection.setAttribute('aria-busy', 'true');
  agentStatus.textContent = mode === 'plan'
    ? 'Running Evidence → Planner → Critic locally. This can take up to a minute…'
    : 'Running the Evidence agent locally…';
  delete agentStatus.dataset.tone;

  try {
    const response = await fetch('/api/agents', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ question: questionField.value.trim(), mode }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
    renderAgentResult(data);
    agentStatus.textContent = data.status === 'answered'
      ? 'Workflow complete. Every statement is tied to cited evidence.'
      : 'Workflow complete. The safety gates withheld a recommendation; see the reason below.';
    agentStatus.dataset.tone = data.status === 'answered' ? 'success' : '';
    if (!agentStatus.dataset.tone) delete agentStatus.dataset.tone;
  } catch (error) {
    agentStatus.textContent = error.message || 'The multi-agent workflow could not complete the request.';
    agentStatus.dataset.tone = 'error';
  } finally {
    agentSubmit.disabled = false;
    agentSection.setAttribute('aria-busy', 'false');
  }
});

updateCount();
