import {copyToClipboard} from './clipboard.mjs';
import {api} from './api.mjs';

const $ = id => document.getElementById(id);
const tabDrafts = new Map();
const editorFields = ['prompt-id','prompt-name','prompt-system','prompt-user','prompt-fact','prompt-answer','prompt-json'];
let prompts = [], category = 'answer', selected = null, editing = false, jsonView = false, changed = () => {};

export function promptSelection() {
  return {answer_prompt_id: $('answer-prompt').value || null,
    review_facts_prompt_id: $('review-facts-prompt').value || null,
    review_answer_prompt_id: $('review-answer-prompt').value || null};
}

export function selectedAnswerPrompt() {
  return prompts.find(prompt => prompt.type === 'answer' && prompt.id === $('answer-prompt').value);
}

function renderRunner() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem('prompt-selection') || '{}'); } catch { /* Ignore invalid preferences. */ }
  for (const [id, type, key, stage] of [
    ['answer-prompt','answer','answer_prompt_id',null],
    ['review-facts-prompt','review','review_facts_prompt_id','facts'],
    ['review-answer-prompt','review','review_answer_prompt_id','answer']]) {
    const select = $(id), previous = select.value || saved[key];
    select.replaceChildren();
    for (const item of prompts.filter(item => item.type === type)) {
      const option = document.createElement('option'); option.value = item.id;
      option.textContent = stage ? `${stage === 'facts' ? 'fact' : 'answer'}_${item.name}` : item.name;
      select.append(option);
    }
    select.value = prompts.some(item => item.id === previous && item.type === type) ? previous : prompts.find(item => item.type === type)?.id || '';
    select.disabled = select.options.length === 0;
  }
  changed();
}

async function reload() { prompts = await api('/prompts'); renderRunner(); }

export async function refreshPrompts() {
  tabDrafts.clear();
  await reload();
  choose();
}

function valueFromForm() {
  if (jsonView) return JSON.parse($('prompt-json').value);
  const value = {format:'traceability-prompt-v1',id:$('prompt-id').value,type:category,
    name:$('prompt-name').value,system_prompt:$('prompt-system').value};
  if (category === 'answer') value.user_prompt = $('prompt-user').value;
  else { value.fact_prompt = $('prompt-fact').value; value.answer_prompt = $('prompt-answer').value; }
  return value;
}

function showValue(value) {
  $('prompt-id').value = value.id || '';
  $('prompt-name').value = value.name || '';
  $('prompt-system').value = value.system_prompt || '';
  $('prompt-user').value = value.user_prompt || '';
  $('prompt-fact').value = value.fact_prompt || '';
  $('prompt-answer').value = value.answer_prompt || '';
  $('prompt-json').value = JSON.stringify(value, null, 2);
}

function render() {
  $('prompt-category-answer').setAttribute('aria-selected', String(category === 'answer'));
  $('prompt-category-review').setAttribute('aria-selected', String(category === 'review'));
  $('prompt-tabs').dataset.active = category;
  for (const type of ['answer','review']) $('prompt-category-'+type).tabIndex = category === type ? 0 : -1;
  $('prompt-editor').setAttribute('aria-labelledby', 'prompt-category-'+category);
  const select = $('managed-prompt'); select.replaceChildren();
  const empty = document.createElement('option'); empty.value = ''; empty.textContent = 'Select a prompt'; select.append(empty);
  for (const item of prompts.filter(item => item.type === category)) {
    const option = document.createElement('option'); option.value = item.id; option.textContent = item.name; select.append(option);
  }
  select.value = selected || '';
  select.disabled = editing;
  $('prompt-editor').hidden = !selected && !editing;
  $('prompt-form-pane').hidden = jsonView;
  $('prompt-json-pane').hidden = !jsonView;
  $('prompt-user-field').hidden = category !== 'answer';
  $('prompt-review-fields').hidden = category !== 'review';
  for (const node of $('prompt-editor').querySelectorAll('input,textarea')) node.readOnly = !editing;
  $('prompt-id').readOnly = !editing || Boolean(selected);
  $('prompt-edit').disabled = editing || !selected;
  $('prompt-delete').disabled = editing || !selected;
  $('prompt-copy').disabled = !selected && !editing;
  $('prompt-save').disabled = !editing;
  $('prompt-new').disabled = editing;
  $('prompt-form-view').setAttribute('aria-pressed', String(!jsonView));
  $('prompt-json-view').setAttribute('aria-pressed', String(jsonView));
  $('prompt-status').textContent = '';
}

function choose(identity) {
  selected = identity || prompts.find(item => item.type === category)?.id || null; editing = false;
  const value = prompts.find(item => item.id === selected);
  if (value) showValue(value);
  render();
}

function switchCategory(type) {
  if (category === type) return;
  tabDrafts.set(category, {selected, editing, jsonView,
    fields: Object.fromEntries(editorFields.map(id => [id, $(id).value]))});
  category = type;
  const draft = tabDrafts.get(type);
  if (draft?.editing) {
    ({selected, editing, jsonView} = draft);
    for (const [id, value] of Object.entries(draft.fields)) $(id).value = value;
    render();
  } else choose(draft?.selected);
}

function switchView(asJSON) {
  const value = valueFromForm();
  if (!value || value.type !== category) throw new Error('Prompt type must match the selected tab.');
  showValue(value); jsonView = asJSON; render();
}

function action(handler) {
  return async event => {
    event?.preventDefault();
    try { await handler(event); } catch (error) { $('prompt-status').textContent = error.message; }
  };
}

export async function installPromptManager(onChange) {
  changed = onChange;
  $('manage-prompts').addEventListener('click', action(async () => {
    await reload(); tabDrafts.clear(); choose(selected); $('prompts-manager').showModal();
  }));
  $('close-prompts').addEventListener('click', () => $('prompts-manager').close());
  $('managed-prompt').addEventListener('change', () => choose($('managed-prompt').value));
  for (const type of ['answer','review']) {
    const tab = $('prompt-category-'+type);
    tab.addEventListener('click', () => switchCategory(type));
    tab.addEventListener('keydown', event => {
      if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
      event.preventDefault();
      const target = event.key === 'Home' ? 'answer' : event.key === 'End' ? 'review' : type === 'answer' ? 'review' : 'answer';
      switchCategory(target); $('prompt-category-'+target).focus();
    });
  }
  $('prompt-new').addEventListener('click', () => {
    selected = null; editing = true;
    showValue({format:'traceability-prompt-v1',id:'prompt-'+crypto.randomUUID(),type:category,name:'',system_prompt:'Required output specification:\n{{output_json}}',
      ...(category === 'answer' ? {user_prompt:'Research question:\n{{user_question}}\n\nDocuments:\n{{documents}}'}
        : {fact_prompt:'{{review_input}}',answer_prompt:'{{review_input}}'})});
    render(); $('prompt-name').focus();
  });
  $('prompt-edit').addEventListener('click', () => {editing=true; render();});
  $('prompt-cancel').addEventListener('click', () => { choose(selected); $('prompts-manager').close(); });
  $('prompt-save').addEventListener('click', action(async () => {
    const value = valueFromForm();
    if (!selected && ['answer','review'].includes(value.type)) category = value.type;
    if (value.type !== category) throw new Error('Prompt type must match the selected tab.');
    if (selected && value.id !== selected) throw new Error('A saved prompt ID cannot change.');
    $('prompt-save').disabled = true;
    try {
      const saved = await api(selected ? `/prompts/${encodeURIComponent(selected)}` : '/prompts', value, selected ? 'PUT' : 'POST');
      await reload(); choose(saved.id); $('prompt-status').textContent = 'Prompt saved.';
    } finally { $('prompt-save').disabled = !editing; }
  }));
  $('prompt-delete').addEventListener('click', action(async () => {
    if (!confirm('Delete this prompt? Recorded runs keep their original prompts.')) return;
    await api(`/prompts/${encodeURIComponent(selected)}`, {} , 'DELETE'); await reload(); choose(null);
  }));
  $('prompt-form-view').addEventListener('click', action(() => switchView(false)));
  $('prompt-json-view').addEventListener('click', action(() => switchView(true)));
  $('prompt-copy').addEventListener('click', action(async () => {
    await copyToClipboard(JSON.stringify(valueFromForm(), null, 2), category === 'answer' ? 'Generation prompt' : 'Review prompt');
    $('prompt-status').textContent = '';
  }));
  for (const id of ['answer-prompt','review-facts-prompt','review-answer-prompt']) $(id).addEventListener('change', () => {
    localStorage.setItem('prompt-selection',JSON.stringify(promptSelection())); changed();
  });
  await reload();
}
