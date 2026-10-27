import test from 'node:test';
import assert from 'node:assert/strict';
import {citationParts, requestSources, requestStatus, evaluationLines, comparisonFields, runMarkdown, propertyRunResults} from '../src/traceability_lab/web/research-view.mjs';

test('property responses preserve expected values and null, zero and false observations', () => {
  for (const value of [null, 0, false]) {
    const batch = {runs:[{id:'saved',status:'completed',answer:'raw'}], evaluations:[{
      automatic:{criteria:[{id:'field',expected:value,observed:value,present:true,result:'pass'}]}
    }]};
    const result = propertyRunResults(batch, 'field');
    assert.equal(result.hasExpected, true);
    assert.equal(result.expected, value);
    assert.equal(result.rows[0].hasObserved, true);
    assert.equal(result.rows[0].observed, value);
    assert.equal(result.rows[0].runId, 'saved');
  }
});

test('missing fields and failed runs retain their saved raw responses and errors', () => {
  const batch = {runs:[{id:'a',status:'completed',answer:'{"other":1}'}, {id:'b',status:'failed',answer:null,error:'Offline'}],
    evaluations:[{automatic:{criteria:[{id:'field',expected:1,observed:null,present:false,result:'fail'}]}}, {automatic:null}]};
  const result = propertyRunResults(batch, 'field');
  assert.equal(result.rows.length, 2);
  assert.equal(result.rows[0].hasObserved, false);
  assert.equal(result.rows[0].rawAnswer, '{"other":1}');
  assert.equal(result.rows[1].result, 'not_evaluated');
  assert.equal(result.rows[1].error, 'Offline');
  assert.equal(result.expectationsVary, true);
});

test('expected output uses frozen Validation JSON, never the Answer JSON structure', () => {
  const batch = {runs:[{id:'a',status:'completed',answer:'Invalid JSON'}], evaluations:[{
    reference:{definition:{answer_json:{answer:'number or null'},validation_json:{answer:42}}},
    automatic:{criteria:[{id:'answer_json',result:'fail'}]}
  }]};
  const result = propertyRunResults(batch, 'answer_json');
  assert.deepEqual(result.expected, {answer:42});
  assert.equal(result.rows[0].rawAnswer, 'Invalid JSON');
  assert.equal(result.format, true);
});

const source = {chunk_id: 'internal-hash', document_id: 'orion', version: '2', chunk_position: 0,
  text: '<script>source data</script>', origin: 'explicit', sha256: 'hash'};
function run(extra = {}) {
  const messages = [{role: 'system', content: 'Exact instructions'}, {role: 'user',
    content: `SOURCES\n[internal-hash] Document: orion; version: 2;\n${source.text}\n\nQUESTION\nWhen?`}];
  return {backend: 'local', status: 'completed', answer: 'Answer [internal-hash] [unknown]',
    context: [source], retrievals: [], claims: [], messages,
    snapshot: {corpus: [], request_body: {messages}, context_mode: 'explicit'}, ...extra};
}

test('readable citations preserve unknown citations and the original answer', () => {
  const stored = run();
  const original = stored.answer;
  const parts = citationParts(stored.answer, stored.context);
  assert.equal(parts.map(part => part.text).join(''), 'Answer [orion v2 · chunk 1] [unknown]');
  assert.equal(parts.find(part => part.source).source.chunk_id, 'internal-hash');
  assert.equal(stored.answer, original);
  assert.equal(citationParts('<img onerror=alert(1)>', [source])[0].text, '<img onerror=alert(1)>');
});

test('configured or selected sources are not enough to establish prompt inclusion', () => {
  assert.deepEqual(requestSources(run({messages: null, snapshot: {}})), []);
  assert.deepEqual(requestSources(run({snapshot: {request_body: {messages: [{content: 'No source context'}]}}})), []);
  assert.equal(requestSources(run()).length, 1);
  assert.match(requestStatus(run({status: 'failed'})), /unconfirmed/);
  assert.match(requestStatus(run({backend: 'demo'})), /no LLM request was sent/);
});

test('evaluation rows describe automated checks without human review rows', () => {
  const evaluation = {automatic: {criteria: [{id:'answer_json',result:'pass'}]}};
  const lines = Object.fromEntries(evaluationLines(run(), evaluation, null));
  assert.match(lines['Evaluation status'], /Per-criterion/);
  assert.ok(Object.keys(lines).every(label => !/Human/.test(label)));
  assert.match(Object.fromEntries(evaluationLines(run(), {}, null))['Evaluation status'], /No automated checks recorded/);
});

test('comparison reflects saved source order, instructions, response and settings', () => {
  const saved = run();
  const other = {...source, chunk_id: 'second', document_id: 'finance'};
  saved.context.push(other);
  saved.snapshot.request_body.messages.at(-1).content += `\n[second] Document: finance; version: 2;\n${other.text}`;
  const first = comparisonFields(saved, {manual_review: null});
  saved.context.reverse();
  const second = comparisonFields(saved, {manual_review: null});
  assert.deepEqual(first['Context in recorded prompt (order)'], second['Context in recorded prompt (order)']);
  assert.deepEqual(first['Context in recorded prompt (order)'].map(s=>s.chunk_id), ['internal-hash','second']);
  assert.equal(first['Raw answer'], saved.answer);
  assert.equal(first['Prompt instructions'].messages[0].content, 'Exact instructions');
  assert.equal(first['Context strategy'], 'explicit');
});

test('comparison distinguishes criterion results from unique trace identifiers', () => {
  const saved = run();
  const a = comparisonFields(saved, {automatic: {run_id: 'a', trace_id: 'trace-a', provenance_uri: '/a',
    evaluator_version: 'v1', criteria: [{id: 'budget', applicability: 'applicable', result: 'pass'}]}});
  const b = comparisonFields(saved, {automatic: {run_id: 'b', trace_id: 'trace-b', provenance_uri: '/b',
    evaluator_version: 'v1', criteria: [{id: 'budget', applicability: 'applicable', result: 'pass'}]}});
  const field = 'Evaluation (automatic results)';
  assert.deepEqual(a[field], b[field]);
  b[field].automatic.criteria[0].result = 'fail';
  assert.notDeepEqual(a[field], b[field]);
});

test('Markdown export keeps the saved question, raw answer and applicability intact', () => {
  const saved = run({id: 'saved-run', trace_id: 'trace-1', question: 'Saved question?', answer: 'Exact answer [internal-hash]'});
  const before = JSON.stringify(saved);
  const evaluation = {automatic: {criteria: [{id: 'budget', applicability: 'undetermined',
    result: 'not_evaluated', reason: 'Question changed.'}]}};
  const markdown = runMarkdown(saved, evaluation);
  assert.match(markdown, /# AI Traceability Lab — run results/);
  assert.match(markdown, /Saved question\?/);
  assert.match(markdown, /Exact answer \[internal-hash\]/);
  assert.match(markdown, /orion v2 · chunk 1/);
  assert.match(markdown, /undetermined/);
  assert.match(markdown, /not_evaluated/);
  assert.match(markdown, /trace-1/);
  assert.equal(JSON.stringify(saved), before);
});

test('Markdown fences preserve raw output containing embedded code fences', () => {
  const saved = run({question: 'Question', answer: 'Before\n```json\n{}\n```\nAfter'});
  assert.ok(runMarkdown(saved, {}).includes('````text\n' + saved.answer + '\n````'));
});

test('Markdown reports distinguish failed and demo runs from successful LLM requests', () => {
  const markdown = runMarkdown(run({question: 'Question', status: 'failed', answer: null, error: 'Connection failed'}), {});
  assert.match(markdown, /receipt is unconfirmed/);
  assert.match(markdown, /No answer saved/);
  assert.match(markdown, /Connection failed/);
  assert.match(runMarkdown(run({question: 'Question', backend: 'demo'}), {}), /no LLM request was sent/);
});


test('combined operator and source context retains only unchanged source evidence', () => {
  const saved = run();
  const message = saved.snapshot.request_body.messages.at(-1);
  message.content = 'ADDITIONAL CONTEXT\nOperator instructions\n\n' + message.content;
  assert.equal(requestSources(saved).length, 1);
  saved.snapshot.context_edited = true;
  assert.equal(requestSources(saved).length, 1);
});

test("analysis export separates failures and includes all raw answers", async () => {
  const {failureMarkdown} = await import("../src/traceability_lab/web/research-view.mjs");
  const run = {id: "a", snapshot: {model: {id: "test-model"}}, backend: "local", batch_id: "batch"};
  const batch = {experiment_title: "Experiment", counts: {total: 2},
    property_counts: [{id: "value", pass: 1, fail: 1}],
    runs: [{id: "a", answer: "PASS_SECRET"}, {id: "b", answer: "wrong"}],
    evaluations: [
      {automatic: {criteria: [{id: "value", result: "pass", expected: 42, observed: "PASS_SECRET"}]}},
      {automatic: {criteria: [{id: "value", result: "fail", expected: 42, observed: 7, reason: "Mismatch"}]}},
    ]};
  const report = failureMarkdown(run, batch);
  assert.ok(report.includes("test-model"));
  assert.ok(report.includes("Run 2"));
  assert.ok(report.indexOf("Expected:") < report.indexOf("Observed:"));
  assert.ok(!report.split("## Raw responses per run")[0].includes("PASS_SECRET"));
  assert.ok(report.split("## Raw responses per run")[1].includes("PASS_SECRET"));
});

test("raw run export preserves whitespace and reports measured duration and usage", async () => {
  const {failureMarkdown} = await import("../src/traceability_lab/web/research-view.mjs");
  const detail = {id: "r", snapshot: {}, status: "completed", answer: '  {"value": 42}\n',
    created_at: "2026-10-21T10:00:00Z", finished_at: "2026-10-21T10:00:02.500Z",
    trace: [{name: "llm.generate", duration_ms: 2200}],
    response: {usage: {completion_tokens: 12, prompt_tokens_details: {cached_tokens: 5}}, choices: [{finish_reason: "stop"}]}};
  const report = failureMarkdown(detail, {runs: [{id: "r"}], counts: {}, property_counts: []}, [detail]);
  assert.ok(report.includes('"duration_ms": 2500'));
  assert.ok(report.includes('"llm_duration_ms": 2200'));
  assert.ok(report.includes('"completion_tokens": 12'));
  assert.ok(report.includes(detail.answer));
  assert.ok(report.includes("1 of 1 measured runs report a positive count"));
});

test("analysis exports frozen validation, evaluator, source hashes and trace times", async () => {
  const {failureMarkdown} = await import("../src/traceability_lab/web/research-view.mjs");
  const run = {id: "r", trace_id: "trace-123", created_at: "2026-10-21T10:00:00Z",
    snapshot: {corpus: [{sha256: "document-hash"}], experiment: {definition: {validation_json: {wrong: "editor"}}}},
    context: [{chunk_id: "source", sha256: "chunk-hash"}]};
  const batch = {runs: [run], counts: {}, evaluations: [{reference: {definition: {validation_json: {value: "FROZEN"}}},
    automatic: {evaluator_version: "defined-json-v2", criteria: []}}]};
  const report = failureMarkdown(run, batch, [run]);
  assert.ok(report.includes('"value": "FROZEN"'));
  assert.ok(!report.includes('"wrong": "editor"'));
  for (const value of ["defined-json-v2", "document-hash", "chunk-hash", "trace-123", run.created_at]) assert.ok(report.includes(value));
});

test("report observations derive dynamic properties, applicability and distinct raw bytes", async () => {
  const {analysisObservations, failureMarkdown} = await import("../src/traceability_lab/web/research-view.mjs");
  const batch = {counts: {total: 999, completed: 999}, runs: [
    {id: "a", status: "completed", answer: "{}"}, {id: "b", status: "completed", answer: " {}"},
    {id: "c", status: "failed", answer: null, error: "offline"}],
    evaluations: [
      {automatic: {criteria: [{id: "arbitrary_property", result: "pass", applicability: "applicable"}]}},
      {automatic: {criteria: [{id: "arbitrary_property", result: "not_evaluated", applicability: "not_applicable"}]}},
      {automatic: null}]};
  const observed = analysisObservations(batch);
  assert.equal(observed.attempted, 3);
  assert.equal(observed.completed, 2);
  assert.equal(observed.failed, 1);
  assert.equal(observed.distinct_raw_responses, 2);
  assert.equal(observed.properties[0].not_evaluated, 2);
  assert.equal(observed.properties[0].pass_percentage_of_attempts, 100 / 3);
  const report = failureMarkdown({snapshot: {}}, batch);
  assert.ok(report.includes("arbitrary\\_property | 1 / 3 | 33.3%"));
  assert.ok(!report.includes("answer_correctness"));
  assert.ok(!report.includes("source_id_correctness"));
  assert.ok(!report.includes("evidence_support"));
});

test("report groups shared references, distinguishes changed references and keeps exact requests", async () => {
  const {failureMarkdown} = await import("../src/traceability_lab/web/research-view.mjs");
  const request = {model: "alias", messages: [{role: "system", content: "  Exact\n"}, {role: "user", content: "Bad JSON?"}],
    temperature: 0, seed: 7, max_tokens: 42, stream: false, response_format: {type: "json_object"}};
  const runs = ["a", "b", "c"].map((id) => ({id, status: "completed", answer: "{invalid", snapshot: {
    context_mode: "explicit", request_body: request, model: {id: "alias", artifact_sha256: null, server_revision: null}}}));
  const evaluation = (value, hash) => ({reference_sha256: hash, reference: {definition: {validation_json: {x: value}}},
    automatic: {evaluator_version: "defined-json-v2", criteria: [{id: "x", method: "deterministic", expected: value, result: "requires_review"}]}});
  const batch = {runs, evaluations: [evaluation(1, "hash-one"), evaluation(1, "hash-one"), evaluation(2, "hash-two")]};
  const before = JSON.stringify({runs, batch});
  const report = failureMarkdown(runs[0], batch, runs);
  const validation = report.split("## Shared frozen validation records")[1].split("## Recorded source identity")[0];
  assert.equal((validation.match(/### Record/g) || []).length, 2);
  assert.ok(validation.includes("runs 1, 2"));
  assert.ok(validation.includes("runs 3"));
  assert.ok(report.includes('"content": "  Exact\\n"'));
  const blocks = [...report.matchAll(/```json\n([\s\S]*?)\n```/g)].map((match) => JSON.parse(match[1]));
  assert.deepEqual(blocks.find((block) => block?.recorded_request)?.recorded_request, request);
  assert.ok(report.includes('"response_format"'));
  assert.ok(report.includes('"seed": 7'));
  assert.ok(report.includes('"sha256": null'));
  assert.ok(report.includes('"revision": null'));
  assert.ok(report.includes("{invalid"));
  assert.equal(JSON.stringify({runs, batch}), before);
});

test("facts report distinguishes semantic observations from deterministic results", async () => {
  const {failureMarkdown, analysisObservations} = await import("../src/traceability_lab/web/research-view.mjs");
  const semantic = {status: "completed", request_body: {model: "evaluator-model"},
    calls: [{phase:"fact",fact_index:0,request_body:{messages:[{role:"user",content:"EXACT FACT EVALUATOR INPUT"}]},raw_answer:"EXACT FACT EVALUATOR OUTPUT"}],
    raw_answer: '{"observation":"raw"}', judgments: {unsupported_fact_indices: [0], expected_facts: [{status: "missing"}], explanation: "Contradiction"}};
  const run = {id: "r", status: "completed", answer: '{"facts":[{"fact":"Claim","source_ids":["s"]}],"answer":null}',
    snapshot: {experiment: {definition: {output_mode: "facts_and_answer"}}}};
  const batch = {runs: [run], evaluations: [{automatic: {criteria: [{id: "source_ids", result: "pass", applicability: "applicable"}], semantic_evaluation: semantic}}]};
  const report = failureMarkdown(run, batch, [run]);
  assert.ok(report.includes("generated facts and answer"));
  assert.ok(report.includes("optional semantic evaluator"));
  assert.ok(report.includes("not deterministic checks or ground truth"));
  assert.ok(report.includes("evaluator-model"));
  assert.ok(report.includes("unsupported_fact_indices"));
  assert.ok(report.includes("EXACT FACT EVALUATOR INPUT"));
  assert.ok(report.includes("EXACT FACT EVALUATOR OUTPUT"));
  assert.deepEqual(analysisObservations(batch).properties.map((p) => p.id), ["source_ids"]);
});

test('facts-mode cards use frozen structure and source rules, not empty exact validation', () => {
  const structure = {facts:[{fact:'string',source_ids:['string']}],answer:'string | null'};
  const batch = {runs:[{id:'r',status:'completed',answer:'{"facts":[],"answer":null}'}],evaluations:[{
    reference:{definition:{output_mode:'facts_and_answer',answer_json:structure,validation_json:{},semantic_reference:{facts:['Expected'],answer:'Expected'}}},
    automatic:{criteria:[{id:'answer_json',result:'pass'},{id:'source_ids',result:'pass',observed:[]}]}
  }]};
  const format = propertyRunResults(batch,'answer_json');
  assert.equal(format.factsMode,true);
  assert.deepEqual(format.expected,structure);
  const sources = propertyRunResults(batch,'source_ids');
  assert.equal(sources.sourceCheck,true);
  assert.equal(sources.hasExpected,true);
  assert.ok(sources.expected.includes('exact match'));
});

test('facts and answer cards use derived combined criteria and display actual values', async () => {
  const {failureMarkdown, analysisObservations} = await import('../src/traceability_lab/web/research-view.mjs');
  const expected = [{fact:'Expected',sources:[{document_id:'orion',version:'1'}]}];
  const observed = [{fact:'Paraphrase',sources:[{document_id:'orion',version:'2'}]}];
  const evaluation = {reference:{definition:{output_mode:'facts_and_answer'}}, automatic:{criteria:[{id:'source_ids',result:'pass'}]}, result_criteria:[{id:'facts',result:'fail',expected,observed},{id:'answer',result:'pass',expected:'Expected answer',observed:'Paraphrase answer'}]};
  const batch = {runs:[{id:'one',status:'completed',answer:'{}'}],evaluations:[evaluation]};
  const facts = propertyRunResults(batch,'facts');
  assert.equal(facts.rows[0].result,'fail');
  assert.deepEqual(facts.expected,expected);
  assert.deepEqual(facts.rows[0].observed,observed);
  assert.equal(propertyRunResults(batch,'answer').rows[0].observed,'Paraphrase answer');
  assert.deepEqual(analysisObservations(batch).properties.map((p) => p.id), ['facts','answer']);
  evaluation.result_criteria_version = 'facts-results-v1';
  batch.runs[0].snapshot = {experiment:{definition:{output_mode:'facts_and_answer'}}};
  const report = failureMarkdown(batch.runs[0], batch, batch.runs);
  assert.ok(report.includes('facts-results-v1'));
  assert.ok(report.includes('"result_criteria"'));
  assert.ok(report.includes('"version": "2"'));
});

test('analysis export includes the same performance overview without modifying it', async () => {
  const {failureMarkdown} = await import('../src/traceability_lab/web/research-view.mjs');
  const run = {id:'r',status:'completed',answer:'{}',snapshot:{}};
  const batch = {runs:[run],evaluations:[],overview:{version:'experiment-performance-v1',total_duration_ms:1500,generator:{input_tokens:20},evaluator:{calls:2},runs:[{run_id:'r'}]}};
  const before = JSON.stringify(batch);
  const report = failureMarkdown(run,batch,[run]);
  assert.ok(report.includes('## Experiment Overview'));
  assert.ok(report.includes('experiment-performance-v1'));
  assert.ok(report.includes('"total_duration_ms": 1500'));
  assert.equal(JSON.stringify(batch),before);
});

test('invalid format rows show one clear reason and preserve the full original response', async () => {
  const {propertyRunResults} = await import('../src/traceability_lab/web/research-view.mjs');
  const raw = '{"fact":"Statement","sources":[';
  const formatReason = 'Invalid JSON. Return one valid JSON object without Markdown or surrounding text.';
  const batch = {runs:[{id:'r',status:'completed',answer:raw}],evaluations:[{
    reference:{definition:{output_mode:'facts_and_answer'}},
    result_criteria:[{id:'answer_json',result:'fail',reason:formatReason},
      {id:'facts',result:'fail',method:'failed_invalid_format',reason:'Review skipped: '+formatReason}]
  }]};
  const row = propertyRunResults(batch,'facts').rows[0];
  assert.equal(row.invalidFormat,true);
  assert.equal(row.displayReason,'Invalid JSON');
  assert.equal(row.rawAnswer,raw);
  batch.runs[0].answer = '{"fact":"Statement"}';
  assert.equal(propertyRunResults(batch,'facts').rows[0].displayReason,'Invalid answer structure');
  assert.equal(row.result,'fail');
  assert.equal(batch.evaluations[0].result_criteria[1].reason,'Review skipped: '+formatReason);
});

test('modern source blocks verify exact grouped text and actual request order', async () => {
  const {contextVerification} = await import('../src/traceability_lab/web/research-view.mjs');
  const chunks = [{...source, text:'First\n', start:0, end:6},
    {...source, chunk_id:'second', text:'part.', start:6, end:11},
    {...source, chunk_id:'finance', document_id:'finance', version:'1', text:'Other budget.', sha256:'kept-hash'}];
  const block = (n, doc, version, text) => `## BEGIN SOURCE ${n}\n\nSource reference: ${JSON.stringify({document_id:doc,version})}\nTitle: "Title"\n\n### BEGIN DOCUMENT\n\`\`\`text\n${text}\n\`\`\`\n### END DOCUMENT\n\n## END SOURCE ${n}`;
  const content = block(1,'finance','1','Other budget.')+'\n\n---\n\n'+block(2,source.document_id,String(source.version),'First\npart.');
  const stored = run({question:'Question?', context:chunks, messages:[{role:'user',content:'wrong fallback'}], snapshot:{context_edited:true, request_body:{messages:[{role:'user',content}]}}});
  assert.deepEqual(requestSources(stored).map(s=>s.chunk_id), ['finance',source.chunk_id,'second']);
  assert.equal(contextVerification(stored).status,'verified');
  assert.equal(requestSources(stored)[0].sha256,'kept-hash');
  stored.snapshot.request_body.messages[0].content = content.replace('Other budget.','Modified budget.');
  assert.equal(contextVerification(stored).status,'partial');
  assert.equal(requestSources(stored).length,2);
  stored.snapshot.request_body.messages[0].content = 'No documents supplied.';
  assert.equal(contextVerification(stored).status,'verification_failed');
  assert.match(runMarkdown(stored,{}),/missing or modified/);
  assert.equal(contextVerification(run({context:[],snapshot:{request_body:{messages:[{role:'user',content:'Question only'}]}}})).status,'no_sources');
  assert.equal(contextVerification(run({messages:null,snapshot:{}})).status,'missing_request');
  assert.equal(contextVerification(run({snapshot:{request_body:{messages:[]}}})).status,'missing_context');
});

test('run report displays temperatures from actual generator and reviewer requests', () => {
  const stored=run({question:'Question?'}); stored.snapshot.request_body.temperature=2;
  stored.snapshot.question_configuration={temperature:0.4};
  const evaluation={automatic:{semantic_evaluation:{configuration:{temperature:0.6},calls:[{request_body:{temperature:0}}]}}};
  const text=runMarkdown(stored,evaluation);
  assert.match(text,/Generator temperature \(recorded request\).*2/);
  assert.match(text,/Reviewer temperature \(recorded requests\).*0/);
});

test('accepted reviewer scores below 100 are highlighted without changing the verdict', () => {
  for (const [score, threshold, result, caution] of [[100,90,'pass',false],[95,90,'pass',true],[90,90,'pass',true],[89,90,'fail',false],[null,null,'pass',false],[95,100,'fail',false]]) {
    const batch={runs:[{id:'run'}],evaluations:[{result_criteria:[{id:'facts',result,confidence:score,confidence_threshold:threshold}]}]};
    const row=propertyRunResults(batch,'facts').rows[0];
    assert.equal(row.caution,caution);
    assert.equal(row.result,result);
  }
});

test('JSON repair is explicit in property rows and both Markdown exports', async () => {
  const {failureMarkdown} = await import('../src/traceability_lab/web/research-view.mjs');
  const stored=run({id:'repaired',question:'Question?',json_repair_status:'repaired'});
  stored.snapshot.json_repair={status:'repaired',original_answer:'{"broken":',request_body:{messages:[{role:'user',content:'repair only'}]},response:{choices:[]}};
  const batch={runs:[stored],evaluations:[{result_criteria:[{id:'answer_json',result:'pass'}]}],counts:{total:1}};
  assert.equal(propertyRunResults(batch,'answer_json').rows[0].jsonRepairStatus,'repaired');
  for (const text of [runMarkdown(stored,{}),failureMarkdown(stored,batch,[stored])]) {
    assert.match(text,/JSON format repair/);
    assert.match(text,/original_answer/);
    assert.match(text,/repair only/);
    assert.match(text,/first-pass/);
  }
});

test('per-fact correctness and confidence stay independent, including legacy and unavailable scores', async () => {
  const {reviewerScores}=await import('../src/traceability_lab/web/research-view.mjs');
  const evaluation={automatic:{semantic_evaluation:{status:'completed',judgments:{facts:[
    {fact_index:1,correctness_score:0,confidence:100,meaning_match:'not_equivalent',source_support:'contradicted'},
    {fact_index:0,correctness_score:95,confidence:90,meaning_match:'equivalent',source_support:'supported'},
    {fact_index:2,confidence:92,meaning_match:'equivalent',source_support:'supported'}],
    answer:{correctness_score:null,confidence:null,meaning_match:'not_evaluated',evidence_support:'not_evaluated'}}}}};
  const original=structuredClone(evaluation);
  const scores=reviewerScores(evaluation);
  assert.deepEqual(scores.map(s=>[s.label,s.correctnessScore,s.confidence]),[
    ['Fact 1',95,90],['Fact 2',0,100],['Fact 3',null,92],['Answer',null,null]]);
  const batch={runs:[{id:'run'}],evaluations:[evaluation]};
  assert.equal(propertyRunResults(batch,'facts').rows[0].reviewerScores.length,3);
  assert.equal(propertyRunResults(batch,'answer').rows[0].reviewerScores.length,1);
  assert.deepEqual(evaluation,original);
  assert.deepEqual(reviewerScores({}),[]);
});

test('both scores appear in Markdown without replacing deterministic outcomes', async () => {
  const {failureMarkdown}=await import('../src/traceability_lab/web/research-view.mjs');
  const stored=run({id:'scores',question:'Question?'});
  const evaluation={result_criteria:[{id:'facts',result:'fail',reason:'Quote is not contiguous.'}],
    automatic:{semantic_evaluation:{judgments:{facts:[{fact_index:0,correctness_score:100,confidence:95}],answer:{confidence:90}}}}};
  const batch={runs:[stored],evaluations:[evaluation],counts:{total:1}};
  for (const report of [runMarkdown(stored,evaluation),failureMarkdown(stored,batch,[stored])]) {
    assert.match(report,/Correctness/);
    assert.match(report,/Confidence/);
    assert.match(report,/100\/100/);
    assert.match(report,/95\/100/);
    assert.match(report,/Unavailable/);
    assert.match(report,/Quote is not contiguous/);
  }
});

test('completed individual review calls remain inspectable when a later review failed', async () => {
  const {reviewerScores}=await import('../src/traceability_lab/web/research-view.mjs');
  const evaluation={automatic:{semantic_evaluation:{status:'failed',calls:[
    {phase:'fact',status:'completed',parsed_answer:{fact_index:0,correctness_score:10,confidence:100}},
    {phase:'fact',status:'failed',parsed_answer:null}]}}};
  assert.deepEqual(reviewerScores(evaluation).map(s=>[s.label,s.correctnessScore,s.confidence]),[['Fact 1',10,100]]);
});

test('mixed batch runs retain their own scores and existing pass/fail results', () => {
  const values=[[95,90,'pass'],[0,100,'fail'],[undefined,92,'pass']];
  const batch={runs:values.map((_,i)=>({id:`run-${i}`})),evaluations:values.map(([correctness_score,confidence,result])=>({
    result_criteria:[{id:'facts',result,confidence,confidence_threshold:90}],
    automatic:{semantic_evaluation:{judgments:{facts:[{fact_index:0,correctness_score,confidence}]}}}}))};
  const rows=propertyRunResults(batch,'facts').rows;
  assert.deepEqual(rows.map(row=>[row.result,row.reviewerScores[0].correctnessScore,row.reviewerScores[0].confidence]),
    [['pass',95,90],['fail',0,100],['pass',null,92]]);
});


import {correctnessSegments} from '../src/traceability_lab/web/research-view.mjs';

test('correctness segments preserve individual scores and interpolate red to green', () => {
  const rows = [{number:1, reviewerScores:[
    {label:'Fact 1', correctnessScore:0, confidence:100},
    {label:'Fact 2', correctnessScore:95, confidence:80},
    {label:'Fact 3', correctnessScore:100, confidence:95}]}];
  const segments = correctnessSegments(rows)[0];
  assert.ok(segments.every(segment => !('text' in segment)));
  assert.equal(segments[0].color, 'rgb(207, 75, 75)');
  assert.equal(segments[2].color, 'rgb(25, 97, 86)');
  assert.equal(segments[1].color, 'rgb(34, 96, 85)');
  assert.match(segments[1].tooltip, /Correctness: 95\/100\nConfidence: 80\/100/);
});

test('run summary averages scores independently and excludes missing values', async () => {
  const {correctnessSummary} = await import('../src/traceability_lab/web/research-view.mjs');
  const scores = [100, 100, 100, 0].map(correctnessScore => ({correctnessScore, confidence:100}));
  assert.equal(correctnessSummary(scores).text, '75');
  assert.match(correctnessSummary(scores).tooltip, /Average confidence: 100/);
  assert.equal(correctnessSummary([{correctnessScore: null, confidence: 80}, {correctnessScore: 0, confidence: 100}]).text, '0');
  assert.match(correctnessSummary([{correctnessScore:null,confidence:80},{correctnessScore:0,confidence:100}]).tooltip, /Average confidence: 90/);
  assert.equal(correctnessSummary([]).text, '—');
  const missing = correctnessSegments([{number:1, reviewerScores:[]}])[0][0];
  assert.equal(missing.color, null);
  assert.match(missing.tooltip, /Correctness: Unavailable/);
});


test('semantic scores and failed citation diagnostics are displayed independently without rewriting records', async () => {
  const {reviewerScores, citationChecks, citationStatus}=await import('../src/traceability_lab/web/research-view.mjs');
  const diagnostic={fact_index:0,document_id:'orion',version:'1',status:'fail',errors:[{code:'whitespace_mismatch',quote_excerpt:'a b',source_excerpt:'a\n\nb',first_difference:{offset:1,quote_character:' ',source_character:'\n'}}]};
  const evaluation={automatic:{criteria:[{id:'source_ids',citation_validations:[diagnostic]}],semantic_evaluation:{judgments:{facts:[{fact_index:0,meaning_match:'equivalent',source_support:'supported',correctness_score:100,confidence:95}]}}}};
  const original=JSON.stringify(evaluation);
  const score=reviewerScores(evaluation)[0];
  assert.deepEqual([score.correctnessScore,score.confidence,score.meaning,score.support],[100,95,'equivalent','supported']);
  assert.equal(citationStatus(score.citations),'Failed');
  const segment=correctnessSegments([{number:1,reviewerScores:[score]}])[0][0];
  assert.equal(segment.citationFailed,true);
  assert.match(segment.tooltip,/Citation validation: Failed/);
  const stored=run({id:'diagnostic',question:'Question?'});
  const {failureMarkdown}=await import('../src/traceability_lab/web/research-view.mjs');
  for (const markdown of [runMarkdown(stored,evaluation), failureMarkdown(stored,{runs:[stored],evaluations:[evaluation],counts:{total:1}},[stored])]) {
    assert.ok(markdown.includes('```json\n'+JSON.stringify([diagnostic],null,2)+'\n```'));
    assert.match(markdown,/equivalent \| supported \| Failed/);
  }
  assert.deepEqual(citationChecks({automatic:{criteria:[]}}),[]);
  assert.equal(citationStatus([]),'Not evaluated');
  assert.equal(citationStatus([{status:'pass'}]),'Verified');
  const batch={runs:[{id:'run',status:'completed',answer:'raw'}],evaluations:[evaluation]};
  const row=propertyRunResults(batch,'facts').rows[0];
  assert.equal(row.citationChecks[0].errors[0].code,'whitespace_mismatch');
  assert.equal(JSON.stringify(evaluation),original);
});


test('human model names are frozen independently from technical API routing', async () => {
  const {modelIdentity,failureMarkdown}=await import('../src/traceability_lab/web/research-view.mjs');
  const stored=run({id:'identity',question:'Question?',snapshot:{...run().snapshot,model:{id:'test-generator-model'},profiles:{llm:{id:'small',name:'Small generator'}}}});
  const evaluation={automatic:{semantic_evaluation:{model:{id:'test-generator-model'},profile:{id:'large',name:'Large reviewer'}}},reproducibility:{configuration_sha256:'config-hash'}};
  const identity=modelIdentity(stored.snapshot.model,stored.snapshot.profiles.llm);
  assert.equal(identity.display_name,'Small generator');
  assert.equal(identity.technical.api_model_id,'test-generator-model');
  assert.equal(modelIdentity({id:'alias-only'}).display_name,'Unknown model');
  assert.equal(modelIdentity({display_name:'Saved name',profile_id:'saved',artifact:{},runtime:{},technical:{api_model_id:'alias'}},{name:'Today’s name'}).display_name,'Saved name');
  const report=runMarkdown(stored,evaluation);
  assert.match(report,/Model:\*\* Small generator/);
  assert.match(report,/Model:\*\* Large reviewer/);
  assert.doesNotMatch(report,/Model:\*\* test-generator-model/);
  assert.match(report,/config-hash/);
  const batch={runs:[stored],evaluations:[evaluation],counts:{total:1}};
  const analysis=failureMarkdown(stored,batch,[stored]);
  assert.match(analysis,/"display_name": "Small generator"/);
  assert.match(analysis,/"api_model_id": "test-generator-model"/);
});

test('reviewer prompt preview replaces only the generated answer in a frozen request', async () => {
  const {reviewerPromptPreview} = await import('../src/traceability_lab/web/research-view.mjs');
  const messages = [{role:'system',content:'Frozen policy. Only schema fields.'},{role:'user',content:JSON.stringify({generated_answer:{facts:[],answer:'ACTUAL OUTPUT'},expected_reference:{answer:'REFERENCE'},output_schema:{type:'object'}})}];
  const evaluation = {automatic:{semantic_evaluation:{calls:[{phase:'answer',request_body:{messages}}]}}};
  const original = JSON.stringify(evaluation);
  const preview = reviewerPromptPreview({},evaluation);
  assert.match(preview.status,/^Recorded/);
  assert.match(preview.text,/Frozen policy/);
  assert.match(preview.text,/__ANSWER_JSON__/);
  assert.match(preview.text,/REFERENCE/);
  assert.doesNotMatch(preview.text,/ACTUAL OUTPUT/);
  assert.equal(JSON.stringify(evaluation),original);
});

test('reviewer preview reconstructs a skipped answer review from frozen evidence and marks it', async () => {
  const {reviewerPromptPreview} = await import('../src/traceability_lab/web/research-view.mjs');
  const run = {question:'Budget?',answer:JSON.stringify({facts:[{sources:[{document_id:'orion',version:'1'}]}],answer:'private generated answer'}),
    context:[{document_id:'orion',version:'1',text:'exact evidence'}, {document_id:'other',version:'1',text:'excluded'}],
    snapshot:{experiment:{semantic_evaluator_snapshot:{policies:{answer:'FROZEN POLICY'},output_schemas:{answer:{type:'object'}}}},evaluation_reference:{definition:{semantic_reference:{answer:'expected'}}}}};
  const before = JSON.stringify(run);
  const preview = reviewerPromptPreview(run,{automatic:{semantic_evaluation:{calls:[{phase:'fact',status:'failed'}]}}});
  assert.match(preview.status,/Reconstructed/);
  assert.match(preview.text,/__ANSWER_JSON__/);
  assert.match(preview.text,/exact evidence/);
  assert.doesNotMatch(preview.text,/private generated answer|excluded/);
  assert.equal(JSON.stringify(run),before);
  assert.match(reviewerPromptPreview({snapshot:{}},{}).status,/No recorded reviewer/);
});

test('compact reviewer preview omits facts, provenance and schema from user input', async () => {
  const {reviewerPromptPreview} = await import('../src/traceability_lab/web/research-view.mjs');
  const run = {question:'Question?',answer:JSON.stringify({facts:[],answer:'generated'}),context:[
    {document_id:'a',version:'1',text:'a\n\n',start:0,end:3,sha256:'do not expose'},
    {document_id:'a',version:'1',text:'b',start:3,end:4},
    {document_id:'uncited',version:'2',text:'all original evidence',start:0,end:21}
  ],snapshot:{experiment:{semantic_evaluator_snapshot:{answer_input_format:'compact-v1',policies:{answer:'Compact frozen instructions'},output_schemas:{answer:{type:'object'}}}},
    evaluation_reference:{definition:{semantic_reference:{answer:'Reference',facts:['PRIVATE EXPECTED FACT']}}}}};
  const preview = reviewerPromptPreview(run,{});
  const payload=JSON.parse(preview.text.split('USER\n')[1]);
  assert.deepEqual(Object.keys(payload),['question','expected_answer','generated_answer','documents']);
  assert.equal(payload.generated_answer,'__ANSWER_JSON__');
  assert.equal(payload.documents[0].text,'a\n\nb');
  assert.equal(payload.documents[1].document_id,'uncited');
  assert.doesNotMatch(preview.text,/PRIVATE EXPECTED FACT|output_schema|fact_judgments|sha256/);
  const frozen = run.snapshot.experiment.semantic_evaluator_snapshot;
  frozen.policies.answer = 'Policy\n{{output_json}}';
  frozen.user_prompts = {answer:'{{review_input}}'};
  frozen.answer_format_instructions = 'Format: required review JSON';
  const templated = reviewerPromptPreview(run,{});
  assert.match(templated.text, /SYSTEM\nPolicy\nFormat: required review JSON/);
  assert.doesNotMatch(templated.text, /\{\{output_json\}\}/);
  assert.equal(templated.text.split('Format:').length - 1, 1);
});

test('evaluation errors remain separate from incorrect answers in report observations', async () => {
  const {analysisObservations}=await import('../src/traceability_lab/web/research-view.mjs');
  const batch={runs:[{id:'r',status:'completed',answer:'raw'}],evaluations:[{result_criteria:[
    {id:'answer',applicability:'applicable',result:'evaluation_error',reason:'Invalid reviewer JSON'}
  ]}]};
  const stats=analysisObservations(batch).properties[0];
  assert.equal(stats.fail,0);
  assert.equal(stats.evaluation_error,1);
});

test('fact reviewer preview selects the recorded individual request without changing evidence', async () => {
  const {reviewerPromptPreview} = await import('../src/traceability_lab/web/research-view.mjs');
  const messages = [{role:'system', content:'Fact policy'}, {role:'user', content:JSON.stringify({generated_fact:'Second fact', supplied_sources:[{text:'Exact evidence\n\n'}]})}];
  const evaluation = {automatic:{semantic_evaluation:{calls:[
    {phase:'fact', fact_index:0, request_body:{messages:[{role:'user',content:'First fact'}]}},
    {phase:'fact', fact_index:1, request_body:{messages}}
  ]}}};
  const original = structuredClone(evaluation);
  const preview = reviewerPromptPreview({}, evaluation, 'fact', 1);
  assert.match(preview.text, /Second fact/);
  assert.match(preview.text, /Exact evidence/);
  assert.doesNotMatch(preview.text, /First fact/);
  assert.deepEqual(evaluation, original);
  assert.equal(reviewerPromptPreview({}, evaluation, 'fact', 2).text, '');
});

test('named review templates preserve instructions while masking only the generated answer', async () => {
  const {reviewerPromptPreview} = await import('../src/traceability_lab/web/research-view.mjs');
  const template = 'Evaluate the answer.\n{{review_input}}\nEnd review data.';
  const payload = {question:'Question?',expected_answer:'Expected',generated_answer:'Actual',documents:[{text:'Evidence'}]};
  const evaluation = {automatic:{semantic_evaluation:{user_prompts:{answer:template},calls:[{
    phase:'answer',request_body:{messages:[{role:'system',content:'Shared policy'},
      {role:'user',content:template.replace('{{review_input}}',JSON.stringify(payload))}]}
  }]}}};
  const before = structuredClone(evaluation);
  const preview = reviewerPromptPreview({},evaluation);
  assert.match(preview.status,/^Recorded/);
  assert.match(preview.text,/Evaluate the answer/);
  assert.match(preview.text,/__ANSWER_JSON__/);
  assert.match(preview.text,/End review data/);
  assert.doesNotMatch(preview.text,/Actual/);
  assert.deepEqual(evaluation,before);
});
