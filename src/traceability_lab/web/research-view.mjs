// Presentation only: stored answers, source text and prompts remain unchanged.

export function modelIdentity(model, profile) {
  if (model?.display_name) return Object.fromEntries(["profile_id", "display_name", "artifact", "runtime", "technical"].map(key => [key, model[key]]));
  const sha = model?.artifact_sha256 || profile?.artifact_sha256 || null;
  return {profile_id: profile?.id ?? null, display_name: profile?.name || "Unknown model",
    artifact: {sha256: sha, identity_status: sha ? "declared" : "unknown", provenance: "historical_snapshot"},
    runtime: {engine: null, revision: model?.server_revision || profile?.server_revision || null},
    technical: {api_model_id: model?.id || profile?.model || null, url: model?.url || profile?.url || null}};
}
function modelMarkdown(role, identity, requests = []) {
  const parameters = requests.filter(Boolean).map(({model, messages, ...parameters}) => parameters);
  return [`## ${role}`, "", `- **Model:** ${markdownText(identity.display_name)}`,
    `- **Profile:** ${markdownText(identity.profile_id)}`, `- **Engine:** ${markdownText(identity.runtime?.engine)}`,
    `- **Engine revision:** ${markdownText(identity.runtime?.revision)}`,
    `- **Artifact filename:** ${markdownText(identity.artifact?.filename)}`,
    `- **Artifact size (bytes):** ${markdownText(identity.artifact?.size_bytes)}`,
    `- **Artifact SHA-256:** ${markdownText(identity.artifact?.sha256)}`,
    `- **Identity:** ${markdownText(identity.artifact?.identity_status)}`,
    `- **Provenance:** ${markdownText(identity.artifact?.provenance)}`, "",
    "Explicit request parameters (omitted parameters use unrecorded server defaults unless observed below):", "", markdownCode(parameters, "json"), "",
    "Field sources distinguish server observations, deployment declarations, local verification and unknowns. Explicit request parameters are listed separately.", "",
    "Declared runtime configuration, observed server defaults and chat template (missing means unknown):", "", markdownCode(identity.runtime, "json"), ""];
}

export const evaluationCriteria = (evaluation) => evaluation?.result_criteria ?? evaluation?.automatic?.criteria ?? [];
export function citationChecks(evaluation) {
  return (evaluation?.automatic?.criteria || []).find(item => item.id === "source_ids")?.citation_validations ?? [];
}
export function citationStatus(checks) {
  return checks.some(check => check.status === "fail") ? "Failed" : checks.length && checks.every(check => check.status === "pass") ? "Verified" : "Not evaluated";
}
export function reviewerScores(evaluation) {
  const observation = evaluation?.automatic?.semantic_evaluation;
  if (!observation) return [];
  const judgments = observation.judgments || {};
  const facts = judgments.facts ?? (observation.calls || [])
    .filter(call => call.phase === "fact" && call.status === "completed" && call.parsed_answer)
    .map(call => call.parsed_answer);
  const answer = judgments.answer ?? judgments.conclusion ?? (observation.calls || [])
    .find(call => ["answer", "conclusion"].includes(call.phase) && call.status === "completed")?.parsed_answer;
  const answerJudgment = answer?.answer ?? answer?.conclusion ?? answer;
  const rows = facts.map(judgment => ({property: "facts", label: `Fact ${judgment.fact_index + 1}`,
    factIndex: judgment.fact_index, judgment})).sort((a,b) => a.factIndex-b.factIndex);
  if (answerJudgment) rows.push({property: "answer", label: "Answer", judgment: answerJudgment});
  return rows.map(({judgment, ...row}) => ({...row,
    correctnessScore: judgment.correctness_score ?? null, confidence: judgment.confidence ?? null,
    meaning: judgment.meaning_match, support: judgment.source_support ?? judgment.evidence_support,
    explanation: judgment.explanation ?? "",
    citations: row.property === "facts" ? citationChecks(evaluation).filter(check => check.fact_index === row.factIndex) : []}));
}

export function correctnessSegments(rows) {
  return rows.map(row => (row.reviewerScores.length ? row.reviewerScores : [{label: "No model judgment", correctnessScore: null, confidence: null}])
    .map(score => {
      const value = score.correctnessScore;
      const color = value == null ? null : `rgb(${Math.round(207 + (25 - 207) * value / 100)}, ${Math.round(75 + (97 - 75) * value / 100)}, ${Math.round(75 + (86 - 75) * value / 100)})`;
      return {color, citationFailed: (score.citations || []).some(check => check.status === "fail"),
        tooltip: `Run ${row.number} · ${score.label}\nCorrectness: ${scoreText(value)}\nConfidence: ${scoreText(score.confidence)}${score.citations?.length ? `\nCitation validation: ${citationStatus(score.citations)}` : ""}`};
    }));
}

export function correctnessSummary(scores) {
  const mean = key => {
    const measured = scores.map(score => score[key]).filter(Number.isFinite);
    return measured.length ? Math.round(measured.reduce((sum, value) => sum + value, 0) / measured.length) : null;
  };
  const correctness = mean("correctnessScore"), confidence = mean("confidence");
  return {text: correctness == null ? "—" : String(correctness),
    tooltip: `Average correctness: ${scoreText(correctness)}
Average confidence: ${scoreText(confidence)}
Scored facts/statements: ${scores.filter(score => Number.isFinite(score.correctnessScore)).length}/${scores.length}`};
}

function scoreText(value) { return value == null ? "Unavailable" : `${value}/100`; }
function scoreMarkdown(evaluation) {
  const rows = reviewerScores(evaluation);
  return ["### Reviewer scores · observations", "",
    "Correctness judges semantic meaning against the reference, independently of citation validity; confidence is certainty in that judgment. Both are subjective, uncalibrated model scores and do not override deterministic failures or define new acceptance thresholds.", "",
    `Recorded reviewer policy: ${evaluation?.automatic?.semantic_evaluation?.prompt_version ?? "Unavailable"}. Historical scores retain the definitions in their recorded prompts.`, "",
    "| Statement | Correctness | Confidence | Semantic meaning | Source support | Citation validation |", "| --- | --- | --- | --- | --- | --- |",
    ...(rows.length ? rows.map(row => `| ${row.label} | ${scoreText(row.correctnessScore)} | ${scoreText(row.confidence)} | ${row.meaning ?? "Unavailable"} | ${row.support ?? "Unavailable"} | ${row.property === "facts" ? citationStatus(row.citations) : "Not applicable"} |`)
      : ["| No model judgment recorded | Unavailable | Unavailable | Unavailable | Unavailable | Not evaluated |"]), "",
    "### Deterministic citation validation", "",
    "Exact, contiguous, character-for-character matching; semantic scores never override a failed citation.", "",
    markdownCode(citationChecks(evaluation), "json"), ""];
}

export function propertyRunResults(batch, propertyId) {
  const rows = (batch.runs || []).map((run, index) => {
    const evaluation = batch.evaluations?.[index];
    const criterion = evaluationCriteria(evaluation).find((item) => item.id === propertyId);
    const format = propertyId === "answer_json";
    const definition = evaluation?.reference?.definition;
    const factsMode = ["facts_and_answer", "facts_and_conclusion"].includes(definition?.output_mode);
    const formatCriterion = evaluationCriteria(evaluation).find((item) => item.id === "answer_json");
    const invalidFormat = factsMode && formatCriterion?.result === "fail";
    let displayReason = criterion?.reason || "No saved criterion result available.";
    if (invalidFormat) {
      try { JSON.parse(run.answer); displayReason = "Invalid answer structure"; }
      catch { displayReason = "Invalid JSON"; }
    }
    const sourceCheck = factsMode && propertyId === "source_ids";
    const validation = factsMode && format ? definition.answer_json : definition?.validation_json;
    const hasExpected = sourceCheck || Boolean(criterion && "expected" in criterion) || (format && validation !== undefined);
    const expected = sourceCheck ? "Every returned source identifier must match a source supplied to this run (exact match)."
      : criterion && "expected" in criterion ? criterion.expected : format ? validation : undefined;
    return {factsMode, sourceCheck, invalidFormat, displayReason, runId: run.id, number: index + 1, status: run.status,
      result: criterion?.result || "not_evaluated", reason: criterion?.reason || "No saved criterion result available.",
      caution: criterion?.result === "pass" && typeof criterion.confidence === "number"
        && typeof criterion.confidence_threshold === "number" && criterion.confidence >= criterion.confidence_threshold && criterion.confidence < 100,
      citationChecks: citationChecks(evaluation),
      reviewerScores: reviewerScores(evaluation).filter(score => score.property === propertyId),
      jsonRepairStatus: run.json_repair_status,
      method: criterion?.method, confidence: criterion?.confidence, confidenceThreshold: criterion?.confidence_threshold,
      hasExpected, expected,
      hasObserved: !format && Boolean(criterion && "observed" in criterion && criterion.present !== false),
      observed: criterion?.observed, rawAnswer: run.answer, error: run.error,
      missing: criterion?.present === false};
  });
  const expectations = new Set(rows.map((row) => row.hasExpected ? JSON.stringify(row.expected) : "not-recorded"));
  return {rows, hasExpected: rows[0]?.hasExpected || false, expected: rows[0]?.expected,
    expectationsVary: expectations.size > 1, factsMode: rows[0]?.factsMode || false, sourceCheck: rows[0]?.sourceCheck || false, format: propertyId === "answer_json"};
}

function markdownText(value) {
  return String(value ?? "Not recorded").replace(/\r?\n/g, " ").replace(/[\\`*_{}\[\]<>|#]/g, "\\$&");
}

function markdownCode(value, language = "text") {
  const text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  const longest = Math.max(2, ...(text.match(/`+/g) || []).map((match) => match.length));
  const fence = "`".repeat(longest + 1);
  return `${fence}${language}\n${text}\n${fence}`;
}

export function runMarkdown(run, evaluation, batch = null) {
  const snapshot = run.snapshot, experiment = snapshot.experiment;
  const temperatures = effectiveTemperatures(run, evaluation);
  const lines = ["# AI Traceability Lab — run results", "", ...[
    ["Run", run.id], ["Created", run.created_at], ["Status", run.status],
    ["Experiment", snapshot.experiment?.definition?.title || snapshot.case?.title || "Custom question"],
    ["Condition", experiment ? `${experiment.modified ? "Modified experiment" : "Original defaults"} · definition ${experiment.definition_version}` : "Historical or custom run"],
    ["Workflow", snapshot.context_mode === "explicit" ? "Controlled experiment · ordered context" : "Exploratory RAG · vector retrieval"],
    ["Model", modelIdentity(snapshot.model, snapshot.profiles?.llm).display_name], ["Backend", run.backend], ["Mode", run.mode],
    ["Repetitions", batch ? `${batch.counts.total} saved runs in batch ${run.batch_id}` : "Single run · no batch recorded"],
    ["Answer prompt", snapshot.experiment?.prompts?.answer?.name || "Historical experiment instructions"],
    ["Review prompt", snapshot.experiment?.prompts?.review?.name || "Built-in reviewer instructions"],
    ["Trace ID", run.trace_id],
    ["Generator temperature (recorded request)", temperatures.generator],
    ["Reviewer temperature (recorded requests)", temperatures.reviewer.length ? temperatures.reviewer.join(", ") : "No reviewer request recorded"],
  ].map(([label, value]) => `- **${label}:** ${markdownText(value)}`), "",
  markdownText(requestStatus(run)), "", "## LLM question", "", markdownCode(run.question), "",
  "## Answer (raw)", "", run.answer == null ? "No answer saved." : markdownCode(run.answer), ""];
  lines.push(...modelMarkdown("Generator", modelIdentity(snapshot.model, snapshot.profiles?.llm), [snapshot.request_body]));
  const reviewer = evaluation.automatic?.semantic_evaluation;
  if (reviewer) lines.push(...modelMarkdown("Semantic reviewer", modelIdentity(reviewer.model, reviewer.profile), reviewer.calls?.map(call => call.request_body) || [reviewer.request_body]));
  lines.push("## Reproducibility", "", markdownCode(evaluation.reproducibility ?? evaluation.automatic?.reproducibility ?? "Not recorded for this historical run.", "json"), "");
  if (snapshot.json_repair) lines.push("## JSON format repair", "",
    "One format-only correction was attempted. The answer above is the effective answer; the original answer and both request/response records are preserved below. Passing after repair is not first-pass format compliance.", "", markdownCode(snapshot.json_repair, "json"), "");
  if (run.error) lines.push("### Run error", "", markdownCode(run.error), "");
  if (run.response?.choices?.[0]?.finish_reason === "length") lines.push("Token limit reached; the answer may be incomplete.", "");
  lines.push("## Evaluation", "");
  for (const criterion of evaluationCriteria(evaluation)) {
    lines.push(`### ${markdownText(criterion.id.replaceAll("_", " "))}`, "",
      `- **Applicability:** ${markdownText(criterion.applicability)}`,
      `- **Result:** ${markdownText(criterion.result)}`,
      `- **Reason:** ${markdownText(criterion.reason)}`);
    for (const key of ["scope", "note", "method"]) if (criterion[key]) lines.push(`- **${key}:** ${markdownText(criterion[key])}`);
    if ("observed" in criterion) lines.push("", markdownCode({observed: criterion.observed, expected: criterion.expected ?? null}, "json"));
    lines.push("");
  }
  if (evaluation.automatic?.semantic_evaluation || citationChecks(evaluation).length) lines.push(...scoreMarkdown(evaluation));
  if (evaluation.automatic?.semantic_evaluation) lines.push("## Optional semantic evaluator", "",
    "Separate model observations, not deterministic checks or ground truth.", "",
    markdownCode(evaluation.automatic.semantic_evaluation, "json"), "");
  for (const [label, value] of evaluationLines(run, evaluation, evaluation.reference)) lines.push(`- **${markdownText(label)}:** ${markdownText(value)}`);
  lines.push("", "## Sources", "", "### Configured versions", "");
  for (const doc of snapshot.corpus || []) lines.push(`- ${markdownText(`${doc.document_id} v${doc.version}`)} · ${markdownText(doc.filename)} · ID: ${markdownText(doc.id)} · SHA-256: ${markdownText(doc.sha256)}`);
  lines.push("", "### Retrieved chunks", "");
  if (!run.retrievals?.length) lines.push("No vector retrieval recorded.");
  for (const source of run.retrievals || []) lines.push(`- ${source.rank}. ${markdownText(source.chunk_id)} · ${markdownText(source.metric)}: ${markdownText(source.distance)}`);
  lines.push("", "### Context in recorded prompt (order)", "");
  const sources = requestSources(run);
  const verification = contextVerification(run);
  const verificationText = {
    verified: "All recorded source fragments verified against exact text and document/version references in the generator request.",
    partial: "Some recorded source fragments are missing or modified in the generator request; only verified fragments are listed.",
    verification_failed: "Source verification failed: recorded source fragments are missing or modified, or their prompt format is unrecognized. This does not establish that no source text was sent.",
    missing_request: "Source verification unavailable: no generator messages were recorded.",
    missing_context: "The recorded generator messages contain no textual user context.",
    no_sources: "No source fragments were recorded for this run; source presence cannot be established from source records.",
  };
  lines.push(verificationText[verification.status]);
  sources.forEach((source, index) => lines.push(`#### ${index + 1}. ${markdownText(sourceLabel(source))}`, "",
    `- **Filename:** ${markdownText(source.filename)}`,
    `- **Chunk ID:** ${markdownText(source.chunk_id)}`,
    `- **SHA-256:** ${markdownText(source.sha256)}`,
    `- **Source span:** ${markdownText(source.start)}–${markdownText(source.end)}`,
    `- **Origin:** ${markdownText(source.origin)}`,
    `- **Retrieval distance:** ${markdownText(source.distance)}`, "", markdownCode(source.text), ""));
  lines.push("", "## Saved configuration", "", markdownCode(snapshot, "json"), "",
    "## Exact recorded request", "", markdownCode(snapshot.request_body ?? {messages: run.messages, note: "Full request body not recorded."}, "json"), "",
    "## Evaluation data", "", markdownCode(evaluation, "json"), "");
  if (batch) lines.push("## Batch totals", "", markdownCode({batch_id: run.batch_id, counts: batch.counts}, "json"), "");
  return lines.join("\n");
}

export function sourceLabel(source) {
  const chunk = source.chunk_position ?? source.position;
  return `${source.document_id} v${source.version} · chunk ${chunk == null ? "?" : chunk + 1}`;
}

export function citationParts(answer, sources) {
  const byId = new Map(sources.map((source) => [source.chunk_id, source]));
  return String(answer || "").split(/(\[[^\]\n]+\])/g).filter(Boolean).map((text) => {
    const source = byId.get(text.slice(1, -1));
    return source && text.startsWith("[") && text.endsWith("]")
      ? {text: `[${sourceLabel(source)}]`, source} : {text};
  });
}

// Verify recorded source bytes against the generator request, never the definition.
export function contextVerification(run) {
  const messages = run.snapshot?.request_body?.messages ?? run.messages;
  if (!Array.isArray(messages)) return {status: "missing_request", sources: []};
  const contexts = messages.filter(message => message.role === "user" || !message.role)
    .map(message => message.content).filter(content => typeof content === "string");
  const expected = run.context || [];
  if (!expected.length) return {status: "no_sources", sources: []};
  if (!contexts.some(content => content.length)) return {status: "missing_context", sources: []};
  const groups = [];
  for (const source of expected) {
    const last = groups.at(-1);
    if (last && last[0].document_id === source.document_id && String(last[0].version) === String(source.version)) last.push(source);
    else groups.push([source]);
  }
  const sources = [], used = new Set();
  const pattern = /^## BEGIN SOURCE (\d+)\n\nSource reference: ([^\n]+)\nTitle: [^\n]*\n\n### BEGIN DOCUMENT\n(`{3,})text\n([\s\S]*?)\n\3\n### END DOCUMENT\n\n## END SOURCE \1(?=\n|$)/gm;
  for (const context of contexts) {
    for (const match of context.matchAll(pattern)) {
      let reference;
      try { reference = JSON.parse(match[2]); } catch { continue; }
      const group = groups.find(items => !used.has(items)
        && items[0].document_id === reference.document_id && String(items[0].version) === reference.version
        && items.reduce((text, item, index) => text + (index && (items[index-1].end === undefined || items[index-1].end !== item.start)
          ? "\n\n[Non-contiguous source excerpt]\n\n" : "") + item.text, "") === match[4]);
      if (group) { sources.push(...group); used.add(group); }
    }
    // Historical prompts used hash headers rather than fenced document blocks.
    if (context.includes("## BEGIN SOURCE ")) continue;
    const legacy = expected.filter(source => !sources.includes(source)).map(source => ({source,
      position: context.indexOf(`[${source.chunk_id}] Document: ${source.document_id}; version: ${source.version};`)}))
      .filter(({source, position}) => {
        const headerEnd = position < 0 ? -1 : context.indexOf("\n", position);
        return headerEnd >= 0 && context.slice(headerEnd + 1).startsWith(source.text);
      })
      .sort((a,b) => a.position-b.position);
    sources.push(...legacy.map(item => item.source));
  }
  return {status: sources.length === expected.length ? "verified" : sources.length ? "partial" : "verification_failed", sources};
}

export function requestSources(run) { return contextVerification(run).sources; }

function effectiveTemperatures(run, evaluation) {
  const reviewer = evaluation?.automatic?.semantic_evaluation;
  return {generator: run.snapshot?.request_body?.temperature ?? "Not recorded",
    reviewer: [...new Set((reviewer?.calls || []).map(call => call.request_body?.temperature).filter(value => value != null))]};
}

export function requestStatus(run) {
  if (run.backend === "demo") return "Demo: prepared context only; no LLM request was sent.";
  if (!run.snapshot.request_body && !run.messages) return "No model request recorded.";
  if (run.status !== "completed") return "Prepared request; model receipt is unconfirmed (run did not complete).";
  return "Context included in the recorded request; the model returned an answer.";
}

export function evaluationLines(run, evaluation, reference) {
  return [
    ["Evaluation status", evaluation.automatic?.criteria?.length ? "Per-criterion checks available; no overall pass/fail criterion defined."
      : "No automated checks recorded; no overall pass/fail criterion defined."],
    ["Expected behavior (frozen reference)", reference?.expected || "No frozen reference defined for this question."],
    ["Observed behavior", "Inspect the saved answer and individual evaluation criteria."],
    ["Automated evaluation", evaluation.automatic?.semantic_evaluation ? "Deterministic validation and separate reviewer-model judgments are recorded. Model judgments are not ground truth." : "No semantic reviewer results recorded. Inspect individual deterministic checks where available."],
  ];
}

export function comparisonFields(run, evaluation) {
  const snapshot = run.snapshot;
  const {run_id, trace_id, provenance_uri, ...automatic} = evaluation.automatic || {};
  return {
    "Experiment condition": snapshot.experiment ?? "Historical or custom run",
    "Question": run.question,
    "Context strategy": snapshot.context_mode || "retrieval",
    "Configured versions (order)": snapshot.corpus,
    "Request status": requestStatus(run),
    "Context in recorded prompt (order)": requestSources(run).map((source) => ({
      label: sourceLabel(source), chunk_id: source.chunk_id, sha256: source.sha256,
      filename: source.filename ?? null, origin: source.origin,
    })),
    "Retrieved chunks (order)": run.retrievals.map(({run_id, ...source}) => source),
    "Model and inference settings": {model: modelIdentity(snapshot.model, snapshot.profiles?.llm),
      configuration: snapshot.question_configuration, backend: run.backend},
    "Prompt instructions": {version: snapshot.prompt_version,
      messages: run.messages?.slice(0, -1), contexts: snapshot.supplemental_contexts},
    "Exact request": snapshot.request_body ?? {messages: run.messages},
    "Raw answer": run.answer,
    "Citations (original IDs)": String(run.answer || "").match(/\[[^\]\n]+\]/g) || [],
    "Evaluation (automatic results)": {...evaluation, automatic: evaluation.automatic ? automatic : null},
  };
}

// Canonical comparison is for grouping records only; exported request bytes stay unchanged.
function recordKey(value) {
  const ordered = (item) => Array.isArray(item) ? item.map(ordered)
    : item && typeof item === "object" ? Object.fromEntries(Object.keys(item).sort().map((key) => [key, ordered(item[key])])) : item;
  return JSON.stringify(ordered(value));
}

export function analysisObservations(batch, details = []) {
  const byId = new Map(details.map((item) => [item.id, item]));
  const runs = (batch.runs || []).map((item) => byId.get(item.id) || item);
  const ids = [...new Set((batch.evaluations || []).flatMap((e) => evaluationCriteria(e).map((c) => c.id)))];
  return {
    attempted: runs.length,
    completed: runs.filter((r) => r.status === "completed").length,
    failed: runs.filter((r) => r.status === "failed").length,
    other_status: runs.filter((r) => !["completed", "failed"].includes(r.status)).length,
    responses_recorded: runs.filter((r) => typeof r.answer === "string").length,
    distinct_raw_responses: new Set(runs.filter((r) => typeof r.answer === "string").map((r) => r.answer)).size,
    properties: ids.map((id) => {
      const criteria = runs.map((_, index) => evaluationCriteria(batch.evaluations?.[index]).find((c) => c.id === id));
      const count = (result) => criteria.filter((c) => c?.result === result).length;
      const passed = count("pass");
      const applicable = criteria.filter((c) => c?.applicability === "applicable").length;
      return {id, attempts: runs.length, pass: passed, fail: count("fail"), evaluation_error: count("evaluation_error"), requires_review: count("requires_review"),
        not_evaluated: criteria.filter((c) => !c || c.result === "not_evaluated").length,
        applicability: {applicable, not_applicable: criteria.filter((c) => c?.applicability === "not_applicable").length,
          undetermined_or_missing: criteria.filter((c) => !c || !["applicable", "not_applicable"].includes(c.applicability)).length},
        pass_percentage_of_attempts: runs.length ? 100 * passed / runs.length : null};
    }),
  };
}

function sharedRecords(lines, title, entries) {
  const groups = new Map();
  entries.forEach((value, index) => {
    const key = recordKey(value);
    if (!groups.has(key)) groups.set(key, {value, runs: []});
    groups.get(key).runs.push(index + 1);
  });
  lines.push(`## ${title}`, "");
  let number = 0;
  for (const group of groups.values()) {
    lines.push(`### Record ${++number} · runs ${group.runs.join(", ")}`, "", markdownCode(group.value, "json"), "");
  }
}

export function failureMarkdown(run, batch, runDetails = []) {
  const snapshot = run.snapshot;
  const byId = new Map(runDetails.map((item) => [item.id, item]));
  byId.set(run.id, byId.get(run.id) || run);
  const details = (batch.runs || []).map((attempt) => ({...attempt, ...(byId.get(attempt.id) || {})}));
  const observations = analysisObservations(batch, details);
  const lines = ["# Experiment Analysis Report v1", "",
    `Experiment: ${markdownText(batch.experiment_title || snapshot.experiment?.definition?.title)}`,
    `Batch: ${markdownText(run.batch_id || batch.batch_id)}`, "",
    "## Observed findings", "",
    `${observations.attempted} attempts; ${observations.completed} completed; ${observations.failed} failed; ${observations.other_status} other statuses.`, "",
    `${observations.distinct_raw_responses} distinct verbatim responses among ${observations.responses_recorded} recorded answers. Missing answers are excluded from this distinct count.`, "",
    "Property | Pass / attempts | Pass % of attempts | Failed | Evaluation errors | Review | Not evaluated | Applicable / not applicable / unknown",
    "--- | ---: | ---: | ---: | ---: | ---: | ---: | ---"];
  for (const p of observations.properties) lines.push(`${markdownText(p.id)} | ${p.pass} / ${p.attempts} | ${p.pass_percentage_of_attempts.toFixed(1)}% | ${p.fail} | ${p.evaluation_error} | ${p.requires_review} | ${p.not_evaluated} | ${p.applicability.applicable} / ${p.applicability.not_applicable} / ${p.applicability.undetermined_or_missing}`);
  lines.push("", "Percentages use all recorded attempts, including failures and unevaluated attempts. A pass applies only to its saved criterion. answer_json checks structure, not factual correctness. Arbitrary property names carry no inferred semantic meaning.", "",
    "### Execution observations", "");
  const latencies = details.flatMap((item) => (item.trace || []).filter((span) => span.name === "llm.generate" && Number.isFinite(span.duration_ms)).map((span) => span.duration_ms));
  lines.push(latencies.length ? `Recorded generator-call latency: min ${Math.min(...latencies).toFixed(1)} ms, max ${Math.max(...latencies).toFixed(1)} ms, mean ${(latencies.reduce((a, b) => a + b, 0) / latencies.length).toFixed(1)} ms (${latencies.length} measurements).` : "Generator-call latency unavailable.", "",
    "Provider usage and timing fields are reproduced per run below. Missing cache statistics mean unknown cache use. These observations do not attribute latency changes to caching.", "",
    "### Limitations", "",
    "Results describe this batch only; repeated runs are not assumed independent. No general model reliability or internal reasoning is established. Matching an identifier does not independently verify evidence support. No additional evidence-support metric is inferred. Missing model hashes, revisions, requests and evaluations remain unknown. Content hashes identify recorded bytes, not human approval.", "");
  const cacheCounts = details.map((item) => item.response?.usage?.prompt_tokens_details?.cached_tokens)
    .filter((value) => Number.isFinite(value) && value >= 0);
  if (cacheCounts.length) lines.push(`Provider-reported cached prompt tokens: ${cacheCounts.filter((n) => n > 0).length} of ${cacheCounts.length} measured runs report a positive count; range ${Math.min(...cacheCounts)}–${Math.max(...cacheCounts)} tokens. ${details.length - cacheCounts.length} runs have no comparable cache measurement.`, "");
  const semanticObservations = (batch.evaluations || []).map((item) => item?.automatic?.semantic_evaluation).filter(Boolean);
  if (semanticObservations.length) lines.push("Optional semantic evaluator observations:", "",
    markdownCode({recorded: semanticObservations.length,
      completed: semanticObservations.filter((item) => item.status === "completed").length,
      failed: semanticObservations.filter((item) => item.status === "failed").length,
      not_evaluated: semanticObservations.filter((item) => item.status === "not_evaluated").length}, "json"), "");
  if (batch.overview) {
    const overview = {...batch.overview}; delete overview.runs;
    lines.push("## Experiment Overview", "", "Persisted execution metrics; generator and evaluator work are separate. Missing values are unknown, not zero.", "", markdownCode(overview, "json"), "");
  }
  sharedRecords(lines, "Model and settings", details.map((item, index) => ({backend: item.backend,
    model: modelIdentity(item.snapshot?.model, item.snapshot?.profiles?.llm),
    reviewer: modelIdentity(batch.evaluations?.[index]?.automatic?.semantic_evaluation?.model, batch.evaluations?.[index]?.automatic?.semantic_evaluation?.profile),
    reproducibility: batch.evaluations?.[index]?.reproducibility ?? null,
    effective_temperatures: effectiveTemperatures(item, batch.evaluations?.[index]),
    settings: item.snapshot?.question_configuration ?? null, prompt_version: item.snapshot?.prompt_version ?? null})));
  lines.push("## Validation JSON and evaluation", "",
    "The following comparison description applies to defined-json-v2 only; other evaluator versions must be interpreted using their recorded rules. Strings compare exactly without trimming, case folding or bracket normalization. Numeric notation may vary; booleans remain distinct from numbers. Empty expected strings disable value checks; null is literal. Nested fields are checked recursively and arrays preserve order. The optional $source rule accepts enclosing brackets before matching source metadata. JSON structure is checked separately against Answer JSON.", "");
  sharedRecords(lines, "Shared frozen validation records", details.map((item, index) => {
    const evaluation = batch.evaluations?.[index];
    const reference = evaluation?.reference ?? item.snapshot?.evaluation_reference ?? null;
    return {validation_json: reference?.definition?.validation_json ?? null,
      evaluator_version: evaluation?.automatic?.evaluator_version ?? null,
      reference_sha256: evaluation?.reference_sha256 ?? evaluation?.automatic?.reference_sha256 ?? null,
      frozen_reference: reference,
      rules: evaluationCriteria(evaluation).map(({id, method, expected}) => ({id, method, expected})) ?? null};
  }));
  sharedRecords(lines, "Recorded source identity", details.map((item) => ({
    corpus_sha256: item.snapshot?.corpus_sha256 ?? null, documents: item.snapshot?.corpus ?? null,
    context_verification: contextVerification(item),
    sources_in_recorded_context: (item.context || []).map(({chunk_id, document_id, version, version_id, sha256, start, end, origin}) =>
      ({chunk_id, document_id, version, version_id, sha256, start, end, origin})),
    context_edited: item.snapshot?.context_edited ?? null, llm_context_sha256: item.snapshot?.llm_context_sha256 ?? null,
    source_tree_sha256: item.snapshot?.source_tree_sha256 ?? null})));
  sharedRecords(lines, "Exact recorded inference requests", details.map((item) => ({
    recorded_request: item.snapshot?.request_body ?? null,
    fallback_messages: item.snapshot?.request_body ? undefined : item.messages ?? null,
    context_mode: item.snapshot?.context_mode ?? null,
    retrieval_parameters: item.snapshot?.context_mode === "explicit"
      ? "top_k and embedding profile are not used for vector retrieval in explicit mode; see recorded trace stages."
      : "See recorded configuration and trace stages.",
    trace_stages: (item.trace || []).map((span) => span.name),
  })));
  lines.push("", "## Failures", "");
  let failures = 0;
  for (const p of observations.properties) {
    const results = propertyRunResults(batch, p.id);
    for (const row of results.rows.filter((r) => r.result === "fail")) {
      failures++;
      lines.push(`### ${markdownText(p.id)} · Run ${row.number}`, "", "Expected:", "",
        row.hasExpected ? markdownCode(row.expected, "json") : "Not recorded.", "",
        "Observed:", "", row.hasObserved ? markdownCode(row.observed, "json")
          : row.missing ? "Field missing." : markdownCode(row.rawAnswer ?? "No answer."), "",
        `Reason: ${markdownText(row.reason)}`, "");
    }
  }
  for (const [index, attempt] of (batch.runs || []).entries()) {
    if (!attempt.error) continue;
    failures++;
    lines.push(`### Run ${index + 1} · execution error`, "", "Expected: completed model response.", "",
      markdownCode(attempt.error), "");
  }
  if (!failures) lines.push("No failed checks or execution errors recorded.");
  lines.push("", "## Raw responses per run", "");
  const detailsById = new Map(runDetails.map((item) => [item.id, item]));
  for (const [index, attempt] of (batch.runs || []).entries()) {
    const detail = detailsById.get(attempt.id) || attempt;
    const elapsed = Date.parse(detail.finished_at) - Date.parse(detail.created_at);
    const generation = detail.trace?.find((span) => span.name === "llm.generate");
    const automatic = batch.evaluations?.[index]?.automatic;
    const {semantic_evaluation: semantic, ...deterministic} = automatic || {};
    const factMode = ["facts_and_answer", "facts_and_conclusion"].includes(detail.snapshot?.experiment?.definition?.output_mode);
    if (factMode) {
      let parsed = null;
      try { parsed = JSON.parse(detail.answer); } catch { /* Invalid raw output is still exported. */ }
      lines.push(`### Run ${index + 1} · generated facts and answer`, "",
        parsed ? markdownCode({facts: parsed.facts, answer: "answer" in parsed ? parsed.answer : parsed.conclusion}, "json") : "Invalid JSON; see raw answer.", "");
    }
    if (factMode || semantic) lines.push(...scoreMarkdown(batch.evaluations?.[index]));
    if (factMode || semantic) lines.push(`### Run ${index + 1} · optional semantic evaluator`, "",
      "Model judgments are separate observations, not deterministic checks or ground truth.", "",
      semantic ? markdownCode(semantic, "json") : "Not configured; no semantic judgments were made.", "");
    lines.push(`### Run ${index + 1}`, "",
      markdownCode({run_id: detail.id, status: detail.status, backend: detail.backend,
        started_at: detail.created_at ?? null, finished_at: detail.finished_at ?? null,
        trace_id: detail.trace_id ?? null,
        reference_sha256: batch.evaluations?.[index]?.reference_sha256 ?? null,
        evaluation: automatic ? deterministic : null,
        result_criteria_version: batch.evaluations?.[index]?.result_criteria_version ?? null,
        result_criteria: batch.evaluations?.[index]?.result_criteria ?? null,
        reference_status: batch.evaluations?.[index]?.reference_status ?? null,
        context_source_integrity: batch.evaluations?.[index]?.context_source_integrity ?? null,
        duration_ms: Number.isFinite(elapsed) && elapsed >= 0 ? elapsed : null,
        llm_duration_ms: generation?.duration_ms ?? null,
        usage: detail.response?.usage ?? null,
        provider_timings: detail.response?.timings ?? null,
        finish_reason: detail.response?.choices?.[0]?.finish_reason ?? null}, "json"), "",
      "Raw answer (verbatim):", "", detail.answer == null ? "No answer recorded." : markdownCode(detail.answer), "");
    lines.push("Recorded response envelope:", "", markdownCode(detail.response ?? null, "json"), "",
      "Trace timings:", "", markdownCode((detail.trace || []).map(({name, span_id, parent_span_id, start_ns, end_ns, duration_ms, status}) =>
        ({name, span_id, parent_span_id, start_ns, end_ns, duration_ms, status})), "json"), "");
    if (detail.snapshot?.json_repair) lines.push("JSON format repair · original answer and correction attempt:", "",
      "These results use the effective answer after one repair attempt; this is separate from first-pass JSON compliance.", "", markdownCode(detail.snapshot.json_repair, "json"), "");
    if (detail.error) lines.push("Error:", "", markdownCode(detail.error), "");
  }
  return lines.join("\n");
}


// Only frozen run data is used; never substitute today's reviewer configuration.
export function reviewerPromptPreview(run, evaluation, phase = "answer", factIndex = 0) {
  const observation = evaluation?.automatic?.semantic_evaluation || {};
  if (phase === "fact") {
    const call = (observation.calls || []).find(item => item.phase === "fact" && item.fact_index === factIndex);
    if (!call?.request_body?.messages) return {status: "No recorded reviewer prompt for this fact. Its review may have been skipped.", text: ""};
    return {
      status: "Recorded fact reviewer prompt, including the generated fact and its cited evidence.",
      text: call.request_body.messages.map(message => `${message.role.toUpperCase()}
${message.content}`).join("\n\n")
    };
  }
  const call = (observation.calls || []).find(item => ["answer", "conclusion"].includes(item.phase));
  const recorded = call?.request_body || (!observation.calls?.length ? observation.request_body : null);
  const render = messages => messages.map(message => `${message.role.toUpperCase()}
${message.content}`).join("\n\n");
  if (recorded?.messages) {
    const messages = structuredClone(recorded.messages);
    let replaced = false;
    for (const message of messages) {
      if (message.role !== "user") continue;
      try {
        const frozen = run.snapshot?.experiment?.semantic_evaluator_snapshot || observation;
        const template = frozen.user_prompts?.answer;
        let payload, wrapped = false;
        if (template?.includes("{{review_input}}")) {
          const [prefix, suffix] = template.split("{{review_input}}");
          if (!message.content.startsWith(prefix) || !message.content.endsWith(suffix)) continue;
          payload = JSON.parse(message.content.slice(prefix.length, suffix ? -suffix.length : undefined));
          wrapped = true;
        } else payload = JSON.parse(message.content);
        if ("generated_answer" in payload) {
          payload.generated_answer = "__ANSWER_JSON__";
          const text = JSON.stringify(payload, null, 2);
          message.content = wrapped ? template.replace("{{review_input}}", () => text) : text;
          replaced = true;
        }
      } catch { /* Keep recorded plain text intact. */ }
    }
    if (replaced) return {status: "Recorded reviewer prompt; generated answer replaced by __ANSWER_JSON__.", text: render(messages)};
  }
  const frozen = run.snapshot?.experiment?.semantic_evaluator_snapshot || observation;
  let policy = frozen.policies?.answer ?? frozen.policies?.conclusion;
  if (policy?.includes("{{output_json}}") && frozen.answer_format_instructions) policy = policy.replace("{{output_json}}", () => frozen.answer_format_instructions);
  const runtime = frozen.model?.runtime || {};
  if (policy && !frozen.user_prompts?.answer && frozen.answer_format_instructions && !(runtime.engine === "llama.cpp" && runtime.field_sources?.engine === "server_observed")) policy += "\n" + frozen.answer_format_instructions;
  const schema = frozen.output_schemas?.answer ?? frozen.output_schemas?.conclusion;
  if (!policy || !schema) return {status: "No recorded reviewer prompt or frozen prompt definition available.", text: ""};
  const definition = run.snapshot?.evaluation_reference?.definition || run.snapshot?.experiment?.definition || {};
  let generated = null;
  try { generated = JSON.parse(run.answer); } catch { /* No invented source selection. */ }
  if (!Array.isArray(generated?.facts)) return {status: "Answer review was not executed; its source context cannot be reconstructed from an invalid answer.", text: policy};
  const cited = generated.facts.flatMap(fact => fact.sources || []);
  const sourceIds = generated.facts.flatMap(fact => fact.source_ids || []);
  const facts = observation.judgments?.facts ?? (observation.calls || []).filter(item => item.phase === "fact" && item.status === "completed" && item.parsed_answer).map(item => item.parsed_answer);
  const compact = frozen.answer_input_format === "compact-v1";
  const documents = [];
  if (compact) {
    if (run.snapshot?.context_edited) return {status: "Answer review was not executed: source context was edited.", text: policy};
    const groups = new Map(), ends = new Map();
    for (const source of run.context || []) {
      const key = JSON.stringify([source.document_id, String(source.version)]);
      if ((!groups.has(key) && (source.start ?? 0) !== 0) || (groups.has(key) && ends.get(key) !== source.start)) {
        return {status: "Answer review was not executed: complete contiguous document text is unavailable.", text: policy};
      }
      if (!groups.has(key)) { const doc = {document_id: source.document_id, version: String(source.version), text: ""}; groups.set(key, doc); documents.push(doc); }
      groups.get(key).text += source.text;
      ends.set(key, source.end);
    }
    for (const doc of documents) {
      const original = definition.documents?.find(item => item.document_id === doc.document_id && String(item.version) === doc.version);
      if (original && original.content !== doc.text) return {status: "Answer review was not executed: document text differs from the frozen original.", text: policy};
    }
  }
  const payload = compact ? {
    question: run.question, expected_answer: definition.semantic_reference?.answer ?? null,
    generated_answer: "__ANSWER_JSON__", documents,
  } : {
    research_question: run.question, generated_answer: "__ANSWER_JSON__", fact_judgments: facts,
    supplied_sources: (run.context || []).filter(source => cited.some(ref => ref.document_id === source.document_id && String(ref.version) === String(source.version)) || sourceIds.includes(source.chunk_id)),
    literal_quote_errors: (evaluation?.automatic?.criteria || []).find(item => item.id === "source_ids")?.quote_errors || [],
    expected_reference: definition.semantic_reference ?? null,
    reference_available: definition.semantic_reference != null,
    source_mapping_verified: !run.snapshot?.context_edited,
    output_schema: schema,
  };
  return {status: "Reconstructed from frozen run data. Answer review was not executed; preceding fact judgments may be incomplete.",
    text: render([{role: "system", content: policy}, {role: "user", content: frozen.user_prompts?.answer ? frozen.user_prompts.answer.replace("{{review_input}}", () => JSON.stringify(payload, null, 2)) : JSON.stringify(payload, null, 2)}])};
}
