import test from 'node:test';
import assert from 'node:assert/strict';
import {blankValidation, validationIsEmpty} from '../src/traceability_lab/web/experiment-editor.mjs';

test('validation skeleton preserves keys and nested structures without answer values', () => {
  const answer = {budget: 2800000, source: 'source ID', details: {known: true, unknown: null}, items: [{name:'text'}]};
  assert.deepEqual(blankValidation(answer), {budget:'', source:'', details:{known:'',unknown:''}, items:[{name:''}]});
  assert.equal(answer.budget, 2800000);
});

test('only an empty draft can be populated automatically', () => {
  for (const text of ['', ' \n', '{}', ' { } ']) assert.equal(validationIsEmpty(text), true);
  for (const text of ['{"budget":2800000}', '{"budget":""}', '{', 'null', '[]']) assert.equal(Boolean(validationIsEmpty(text)), false);
});



test('facts output template and legacy definition defaults stay generic', async () => {
  const {factAnswerTemplate, definitionDefaults} = await import('../src/traceability_lab/web/experiment-editor.mjs');
  assert.deepEqual(Object.keys(factAnswerTemplate), ['facts', 'answer']);
  assert.match(factAnswerTemplate.facts[0].sources[0].quote, /exact, contiguous quotation/);
  const legacy = {title:'Legacy', answer_json:{custom:'value'}};
  const normalized = definitionDefaults(legacy);
  assert.equal(normalized.output_mode, 'custom');
  assert.equal(normalized.semantic_reference, null);
  assert.equal(normalized.semantic_evaluator, null);
  assert.deepEqual(normalized.answer_json, legacy.answer_json);
  assert.equal('output_mode' in legacy, false);
});

test('experiment JSON keeps documents while omitting prompts and generated output configuration', async () => {
  const {experimentJSONValue, parseExperimentJSON} = await import('../src/traceability_lab/web/experiment-editor.mjs');
  const definition = {title:'Fixture', question:'Question?', system_prompt:'My policy', documents:[], output_mode:'facts_and_answer', version_ids:[], answer_json:{generated:true}, validation_json:{}, semantic_reference:{facts:[],answer:null}};
  const value = experimentJSONValue(definition);
  assert.equal('system_prompt' in value, false);
  assert.deepEqual(value.documents, []);
  assert.equal('output_mode' in value, false);
  assert.equal('answer_json' in value, false);
  assert.equal('version_ids' in value, false);
  assert.equal(definition.output_mode, 'facts_and_answer');
  const parsed = parseExperimentJSON(JSON.stringify(value));
  assert.equal(parsed.output_mode, 'facts_and_answer');
  assert.equal(parsed.system_prompt, '');
  assert.deepEqual(parseExperimentJSON(JSON.stringify({format:'traceability-experiment-v2',experiment:value})), parsed);
  for (const text of ['{','[]','null','{"format":"traceability-experiment-v2","experiment":[]}']) assert.throws(() => parseExperimentJSON(text));
});


test('fixed output changes do not block saved experiments but actual edits still do', async () => {
  const {experimentChanged, factAnswerTemplate, experimentJSONValue, parseExperimentJSON} = await import('../src/traceability_lab/web/experiment-editor.mjs');
  const saved = {title:'Baseline', question:'Who?', context:'{{user_question}}', documents:[{document_id:'orion', version:'1', enabled:true}], version_ids:[], output_mode:'facts_and_answer', answer_json:{facts:[], answer:'string | null'}, validation_json:{}};
  const draft = {...saved, answer_json:factAnswerTemplate};
  assert.equal(experimentChanged(saved, draft), false);
  assert.equal(experimentChanged(saved, parseExperimentJSON(JSON.stringify(experimentJSONValue(draft)))), false);
  for (const change of [{title:'Changed'}, {question:'Different?'}, {documents:[]}, {semantic_reference:{facts:[],answer:null}}]) {
    assert.equal(experimentChanged(saved, {...draft, ...change}), true);
  }
  const custom = {...saved, output_mode:'custom', answer_json:{answer:'string'}};
  assert.equal(experimentChanged(custom, {...custom, system_prompt:'', context:''}), false);
  assert.equal(experimentChanged(custom, {...custom, answer_json:{answer:'number'}}), true);
});

test('old default threshold migrates once while preserving other reviewer settings', async () => {
  const {reviewSettings} = await import('../src/traceability_lab/web/experiment-editor.mjs');
  const old={llm_profile_id:'large',temperature:0,max_tokens:4096,confidence_threshold:100};
  assert.deepEqual(reviewSettings(old,false),{...old,confidence_threshold:90});
  assert.equal(old.confidence_threshold,100);
  assert.equal(reviewSettings(old,true).confidence_threshold,100);
  assert.equal(reviewSettings({...old,confidence_threshold:80},false).confidence_threshold,80);
});

test('run token override takes priority without changing the saved profile', async () => {
  const {outputTokenLimit} = await import('../src/traceability_lab/web/experiment-editor.mjs');
  const profile = {max_tokens: 2048};
  assert.equal(outputTokenLimit('4096', profile, 1024), 4096);
  assert.equal(outputTokenLimit('', profile, 1024), 2048);
  assert.equal(outputTokenLimit('', null, 1024), 1024);
  assert.equal(outputTokenLimit('', {}, 4096), 4096);
  assert.deepEqual(profile, {max_tokens: 2048});
});
