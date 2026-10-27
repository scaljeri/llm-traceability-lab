import test from 'node:test';
import assert from 'node:assert/strict';
import {fieldHelp} from '../src/traceability_lab/web/field-help.mjs';

const field = (id, extras = {}) => ({id, dataset:{}, labels:[], getAttribute:() => null, closest:() => null, ...extras});

test('runner help explains sampling, independent review and failure thresholds', () => {
  assert.match(fieldHelp(field('seed')).text, /does not guarantee identical/);
  assert.match(fieldHelp(field('seed-enabled')).text, /no seed is sent/);
  assert.match(fieldHelp(field('semantic-evaluator')).text, /independent model/);
  assert.match(fieldHelp(field('evaluator-threshold')).text, /not proof/);
  assert.match(fieldHelp(field('evaluator-max-tokens')).text, /per review response, not per batch/);
});

test('dynamic document fields explain identity, body and availability', () => {
  assert.match(fieldHelp(field('', {dataset:{field:'document_id'}})).text, /Together with version/);
  assert.match(fieldHelp(field('', {dataset:{field:'content'}})).text, /Disabled documents/);
  assert.match(fieldHelp(field('', {type:'checkbox', closest:() => ({})})).text, /excludes it/);
});

test('temperature help states the allowed range and decimal examples for both models', () => {
  for (const id of ['temperature', 'evaluator-temperature']) {
    const text = fieldHelp(field(id)).text;
    assert.match(text, /0–2/);
    assert.match(text, /0\.4/);
    assert.match(text, /not a percentage/);
  }
});

test('prompt field help explains only the wildcards available in that stage', () => {
  const answer = fieldHelp(field('prompt-user')).text;
  for (const slot of ['{{user_question}}', '{{documents}}', '{{output_json}}']) assert.ok(answer.includes(slot));
  assert.match(answer, /configured order/);
  assert.match(answer, /system prompt/);
  for (const id of ['prompt-fact', 'prompt-answer']) {
    const help = fieldHelp(field(id)).text;
    assert.match(help, /\{\{review_input\}\}/);
    assert.match(help, /exactly once/);
  }
  assert.match(fieldHelp(field('prompt-system')).text, /\{\{output_json\}\}/);
  assert.match(fieldHelp(field('prompt-system')).text, /Expected answers are never inserted/);
});
