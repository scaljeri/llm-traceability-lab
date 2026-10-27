import {copyToClipboard} from "./clipboard.mjs";
import {installPromptManager, promptSelection, selectedAnswerPrompt, refreshPrompts} from "./prompt-manager.mjs";
import {api} from "./api.mjs";
import {installFieldHelp} from "./field-help.mjs";
// Render source and model text as plain text, never as executable HTML.
import {runMarkdown, failureMarkdown, propertyRunResults, correctnessSegments, correctnessSummary, citationStatus, reviewerPromptPreview} from "./research-view.mjs";
import {outputTokenLimit, blankValidation, validationIsEmpty, factAnswerTemplate, definitionDefaults, experimentJSONValue, parseExperimentJSON, experimentChanged, reviewSettings} from "./experiment-editor.mjs";
const $ = (id) => document.getElementById(id);
let contextPreviewError = "";
let contextPreview = null, previewKey = "", previewTimer, previewSequence = 0;
let currentRunMarkdown = "", currentProvenance = null, reportSequence = 0;
let currentRun = null, serviceProfiles = [];
let experimentDefinitions = [], editingExperiment = null, experimentDocuments = [];
let runningExperiment = false;
let generatedValidation = null;

function showError(error) {
  const menu = $("settings-menu");
  if ($("experiments-manager").open && !menu.open) $("experiments-manager").querySelector(".manager-body").prepend($("error"));
  else if (menu.open) menu.querySelector(".manager-body").prepend($("error"));
  else document.querySelector("main").before($("error"));
  $("error").textContent = error.message;
  $("error").hidden = false;
}

function guarded(action) {
  return async (event) => {
    event?.preventDefault();
    $("error").hidden = true;
    const button = event?.submitter;
    if (button) button.disabled = true;
    try { await action(event); } catch (error) {
      if (event?.target.id === "profile-form") $("profile-status").textContent = `Save failed: ${error.message}`;
      showError(error);
    }
    finally { if (button) button.disabled = button.id === "save-profile" && !profileEditing; }
  };
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function option(select, value, text) {
  const node = element("option", text); node.value = value; select.append(node);
}

function json(id, value) { $(id).textContent = JSON.stringify(value, null, 2); }
function characters(text) { return Array.from(text || ""); }

async function loadLLMProfiles() { return (await api("/profiles")).filter(profile => profile.kind === "llm"); }

function renderProfiles(initial = false) {
  for (const [id, storage] of [["llm-profile", "answer-model"], ["semantic-evaluator", null]]) {
    const select = $(id), previous = initial && storage ? localStorage.getItem(storage) : select.value;
    select.replaceChildren();
    for (const profile of serviceProfiles) option(select, profile.id, profile.name);
    select.value = serviceProfiles.some(profile => profile.id === previous) ? previous : (serviceProfiles[0]?.id || "");
    if (storage) localStorage.setItem(storage, select.value);
  }
  if (initial) {
    try { const config = JSON.parse(localStorage.getItem("review-settings") || "null"); if (config) setEvaluatorConfig(config); }
    catch { localStorage.removeItem("review-settings"); }
  }
  const selected = $("edit-profile").value;
  $("edit-profile").replaceChildren(); option($("edit-profile"), "", "Choose a server");
  for (const profile of serviceProfiles) option($("edit-profile"), profile.id, `${profile.kind} · ${profile.name}`);
  $("edit-profile").value = selected;
  $("close-settings").disabled = serviceProfiles.length === 0;
  scheduleContextPreview();
}

// A result belongs to the exact profile configuration checked in this page session.
let profileEditing = false;
function setProfileEditing(editing) {
  profileEditing = editing;
  const selected = serviceProfiles.some(profile => profile.id === $("edit-profile").value);
  $("profile-management").hidden = !selected && !editing;
  $("profile-empty").hidden = selected || editing;
  for (const control of $("profile-form").querySelectorAll("input, select, textarea, button")) {
    control.disabled = !editing;
  }
  $("profile-kind").disabled = !editing || selected;
  $("save-profile").disabled = !editing;
  $("edit-server").disabled = !selected || editing;
  $("delete-profile").disabled = !selected || editing;
  $("edit-profile").disabled = editing;
  $("new-profile").disabled = editing;

}
$("new-profile").addEventListener("click", () => {
  profileEditing = true;
  $("edit-profile").value = "";
  $("edit-profile").dispatchEvent(new Event("change"));
  $("profile-name").focus();
});
$("edit-server").addEventListener("click", () => setProfileEditing(true));
$("cancel-profile").addEventListener("click", () => {
  profileEditing = false;
  $("edit-profile").dispatchEvent(new Event("change"));
  if (!serviceProfiles.length) $("new-profile").click();
});
$("edit-profile").addEventListener("change", () => {
  const profile = serviceProfiles.find((item) => item.id === $("edit-profile").value);
  $("profile-form").reset(); $("profile-kind").disabled = Boolean(profile);
  $("profile-model").replaceChildren();
  option($("profile-model"), profile?.model || "", profile?.model || "Load models to select");
  $("profile-model-status").textContent = "";
  if (profile) {
    for (const [id, key] of [["name", "name"], ["kind", "kind"], ["url", "url"], ["model", "model"],
      ["hardware", "hardware"], ["precision", "precision"], ["revision", "revision"], ["server-revision", "server_revision"], ["sha", "artifact_sha256"],
      ["max-tokens", "max_tokens"], ["timeout", "timeout_seconds"], ["thinking", "enable_thinking"]]) {
      $(`profile-${id}`).value = profile[key] == null ? "" : String(profile[key]);
    }
  }
  $("profile-manifest").value = profile?.deployment_manifest ? JSON.stringify(profile.deployment_manifest, null, 2) : "";
  $("profile-capture-metadata").value = profile?.capture_server_metadata == null ? "" : String(profile.capture_server_metadata);
  $("profile-trust-identity").value = String(profile?.trust_server_identity || false);
  setProfileEditing(!profile && profileEditing);
   $("profile-status").textContent = ""; $("profile-verification-status").textContent = "";
});
$("load-profile-models").addEventListener("click", guarded(async () => {
  const input = $("profile-url");
  if (!input.reportValidity()) return;
  const button = $("load-profile-models");
  button.disabled = true; $("profile-model-status").textContent = "Loading models…";
  try {
    const result = await api("/profile-models", {name: "Model discovery", kind: "llm", url: input.value, model: "discovery"});
    if (!result.reachable) throw new Error(result.error || "Unable to load models.");
    const select = $("profile-model"), previous = select.value;
    select.replaceChildren();
    for (const id of result.models) option(select, id, id);
    if (result.models.includes(previous)) select.value = previous;
    $("profile-model-status").textContent = result.models.length ? `${result.models.length} models available.` : "No models returned by the server.";
  } catch (error) {
    $("profile-model-status").textContent = error.message;
  } finally { button.disabled = !profileEditing; }
}));
$("delete-profile").addEventListener("click", guarded(async () => {
  const id = $("edit-profile").value, profile = serviceProfiles.find(item => item.id === id);
  if (!profile || !window.confirm(`Delete server “${profile.name}”? Recorded runs are preserved.`)) return;
  await api(`/profiles/${id}`, {}, "DELETE");
  serviceProfiles = await loadLLMProfiles(); renderProfiles();
  $("edit-profile").value = ""; profileEditing = false;
  $("edit-profile").dispatchEvent(new Event("change"));
  if (!serviceProfiles.length) openServerManager();
}));
$("verify-profile-artifact").addEventListener("click", guarded(async () => {
  const id = $("edit-profile").value;
  const saved = serviceProfiles.find(profile => profile.id === id);
  if (!saved) throw new Error("Save or select a server profile before verifying an artifact.");
  if (saved.url !== $("profile-url").value || saved.model !== $("profile-model").value) throw new Error("Save profile changes before verifying the artifact.");
  const path = $("profile-artifact-path").value.trim();
  if (!path) throw new Error("Enter the GGUF path on the application host.");
  const button = $("verify-profile-artifact"); button.disabled = true;
  $("profile-verification-status").textContent = "Hashing the complete GGUF file… This can take time for large models.";
  try {
    const profile = await api(`/profiles/${id}/verify-artifact`, {path, pid: $("profile-artifact-pid").value ? Number($("profile-artifact-pid").value) : null});
    serviceProfiles = await loadLLMProfiles(); renderProfiles();
    $("edit-profile").value = id; $("edit-profile").dispatchEvent(new Event("change"));
    $("profile-verification-status").textContent = `File hash registered: ${profile.deployment_manifest.sha256} · ${profile.deployment_manifest.size_bytes} bytes. Loaded-artifact verification is checked separately at run start.${profile.local_artifact_verification?.association_error ? ` Association unavailable: ${profile.local_artifact_verification.association_error}` : ""}`;
  } catch (error) {
    $("profile-verification-status").textContent = error.message; throw error;
  } finally { button.disabled = false; }
}));
$("profile-form").addEventListener("submit", guarded(async () => {
  $("profile-status").textContent = "Saving…";
  const id = $("edit-profile").value, kind = $("profile-kind").value;
  const data = {deployment_manifest: $("profile-manifest").value.trim() ? JSON.parse($("profile-manifest").value) : null,
    capture_server_metadata: $("profile-capture-metadata").value === "" ? null : $("profile-capture-metadata").value === "true", trust_server_identity: $("profile-trust-identity").value === "true",
    name: $("profile-name").value, kind, url: $("profile-url").value, model: $("profile-model").value,
    hardware: $("profile-hardware").value || null, precision: $("profile-precision").value || null,
    revision: $("profile-revision").value || null, artifact_sha256: $("profile-sha").value || null,
    server_revision: $("profile-server-revision").value || null,
    max_tokens: kind === "llm" && $("profile-max-tokens").value !== "" ? Number($("profile-max-tokens").value) : null,
    timeout_seconds: Number($("profile-timeout").value),
    enable_thinking: kind === "llm" && $("profile-thinking").value !== "" ? $("profile-thinking").value === "true" : null};
  const saved = await api(id ? `/profiles/${id}` : "/profiles", data, id ? "PUT" : "POST");
  serviceProfiles = await loadLLMProfiles(); renderProfiles();
  $("edit-profile").value = saved.id; $("edit-profile").dispatchEvent(new Event("change"));
  $(`${kind}-profile`).value = saved.id;
  scheduleContextPreview();
  $("profile-status").textContent = "Server saved.";
}));

function showTemperature(id) {
  $(`${id}-value`).value = Number($(id).value).toFixed(1);
}
for (const id of ["temperature", "evaluator-temperature"]) {
  $(id).addEventListener("input", () => showTemperature(id));
  showTemperature(id);
}

function temperatureValue(id) {
  const input = $(id), value = Number(input.value);
  const name = id === "temperature" ? "Temperature" : "Review temperature";
  const message = input.value.trim() && Number.isFinite(value) && value >= 0 && value <= 2
    ? "" : `${name} must be between 0 and 2. Use e.g. 0.8, not 80 (not a percentage).`;
  input.setCustomValidity(message);
  input.setAttribute("aria-invalid", String(Boolean(message)));
  if (message) throw new Error(message);
  return value;
}

function serviceSelection() {
  const llm = $("llm-profile").value || null;
  return {context_ids: [],
    llm_profile_id: llm, temperature: temperatureValue("temperature"),
    seed: $("seed-enabled").checked ? Number($("seed").value) : null, max_tokens: outputTokenLimit($("max-tokens").value, serviceProfiles.find((item) => item.id === llm), 1024)};
}


function renderOwnedSourceIds() {
  for (const row of $("owned-documents").children) {
    const target = row.querySelector(".owned-source-ids"), doc = row.ownedDocument;
    if (!target || !doc) continue;
    target.replaceChildren();
    const copy = element("button", "Copy source JSON", "secondary");
    copy.type = "button"; copy.title = "Copy source JSON to clipboard";
    copy.addEventListener("click", guarded(async () => {
      await copyToClipboard(JSON.stringify({document_id: doc.document_id, version: doc.version}, null, 2), "Source JSON");
    }));
    target.append(copy);
  }
}
function renderOwnedDocuments() {
  const list = $("owned-documents"); resetDragList(list); list.replaceChildren();
  for (const [index, doc] of experimentDocuments.entries()) {
    const row = element("article", undefined, "owned-document"); row.ownedDocument = doc; row.dataset.key = `owned-${index}`;
    const heading = element("div", undefined, "heading");
    const grip = dragHandle(row, list, `${doc.document_id} v${doc.version}`, () => {
      experimentDocuments = [...list.children].map((item) => item.ownedDocument);
      updateExperimentState();
    }); grip.disabled = !managerEditing;
    const enabled = document.createElement("input"); enabled.type = "checkbox"; enabled.setAttribute("aria-label", `Enable ${doc.document_id} v${doc.version}`); enabled.checked = doc.enabled; enabled.disabled = runningExperiment;
    enabled.addEventListener("change", () => { doc.enabled = enabled.checked; if (!managerEditing) setExperimentEditing(true); else row.classList.toggle("document-disabled", !doc.enabled); updateExperimentState(); });
    const label = element("label", undefined, "check"); label.append(enabled, document.createTextNode("Enabled"));
    const remove = element("button", "Remove", "secondary"); remove.type = "button"; remove.disabled = runningExperiment;
    remove.addEventListener("click", () => { experimentDocuments = experimentDocuments.filter((item) => item !== doc); if (!managerEditing) setExperimentEditing(true); else renderOwnedDocuments(); updateExperimentState(); });
    heading.append(grip, label, remove); row.append(heading);
    row.classList.toggle("document-disabled", !doc.enabled);
    const fields = element("details"), summary = element("summary", `${doc.document_id} · v${doc.version} · ${doc.filename || "Document text"}`);
    fields.append(summary); fields.open = !doc.content;
    row.append(fields);
    for (const [key, title] of [["document_id", "Document ID"], ["version", "Version"], ["filename", "Filename"], ["title", "Title"], ["content", "Document text"]]) {
      const fieldLabel = element("label", title);
      const field = document.createElement(key === "content" ? "textarea" : "input");
      field.value = doc[key] || ""; field.readOnly = !managerEditing;
      field.dataset.field = key; field.setAttribute("aria-label", title);
      if (key === "content") field.rows = 6;
      field.addEventListener("input", () => { doc[key] = field.value; summary.textContent = `${doc.document_id} · v${doc.version} · ${doc.filename || "Document text"}`; updateExperimentState(); });
      fieldLabel.append(field); fields.append(fieldLabel);
    }
    row.append(element("div", undefined, "owned-source-ids")); list.append(row);
  }
  if (!experimentDocuments.length) list.append(element("p", "No documents. Add a document or import an experiment.", "muted"));
  renderOwnedSourceIds();
}
$("add-owned-document").addEventListener("click", () => {
  if (!managerEditing) setExperimentEditing(true);
  let number = 1;
  while (experimentDocuments.some((doc) => doc.document_id === `document-${number}`)) number++;
  experimentDocuments.push({document_id: `document-${number}`, version: "1", filename: "", title: "", source: "", content: "", metadata: {}, enabled: true});
  renderOwnedDocuments(); updateExperimentState();
});
function researchSelection() {
  return {question: $("question").value, documents: experimentDocuments,
    context_mode: "explicit"};
}

function updateExperimentState() {
  scheduleContextPreview();
  const saved = experimentDefinitions.find((item) => item.id === editingExperiment);
  let changed = true;
  try { changed = !saved || experimentChanged(saved.definition, definitionFromEditor()); }
  catch { /* Keep incomplete JSON editable; saving/running reports the error. */ }
  const missingPrompt = ["answer-prompt", "review-facts-prompt", "review-answer-prompt"].some(id => !$(id).value);
  $("run-button").disabled = runningExperiment || missingPrompt || !saved || changed || !contextPreview || previewKey !== contextConfigurationKey();
  $("runner-selection-status").textContent = !saved ? "Select a saved experiment, or open the editor to create one."
    : missingPrompt ? "Choose a generation prompt, a fact review prompt and an answer review prompt. Create missing prompts in Manage prompts."
    : missingDefinitionSources.length ? "A saved document is unavailable. Update the selection in the editor."
    : contextPreviewError ? contextPreviewError
    : changed ? "Save your experiment changes before starting."
    : `${saved.definition.title} · ${experimentDocuments.filter((doc) => doc.enabled).length} enabled documents · ${contextPreview && previewKey === contextConfigurationKey() ? "ready" : "preparing input…"}`;
}

function refreshResearchConfiguration() {
  updateExperimentState();
}

document.addEventListener("input", (event) => {
  if (event.target.matches("#output-mode, #semantic-reference, #semantic-evaluator, #evaluator-temperature, #evaluator-seed, #evaluator-max-tokens, #experiment-title, #answer-json, #validation-json, #question, #temperature, #seed, #max-tokens, #llm-profile")) updateExperimentState();
});

async function loadRun(id) {
  const sequence = ++reportSequence;
  $("batch-run-selection").value = id;
  $("run-panel").hidden = true;
  $("copy-markdown").disabled = $("download-markdown").disabled = $("copy-provenance").disabled = true;
  currentRunMarkdown = ""; currentProvenance = null;
  $("markdown-copy-status").textContent = $("provenance-copy-status").textContent = "";
  $("report-status").textContent = "Preparing reports…";
  try {
    const [run, provenance, evaluation] = await Promise.all([
      api(`/runs/${id}`), api(`/runs/${id}/provenance`), api(`/runs/${id}/evaluation`),
    ]);
    const batch = exportBatch?.runs.some((item) => item.id === id) ? exportBatch
      : run.batch_id ? await api(`/batches/${run.batch_id}`) : null;
    if (sequence !== reportSequence) return;
    currentRun = run; currentProvenance = provenance;
    currentRunMarkdown = runMarkdown(run, evaluation, batch);
    $("download").href = `/api/runs/${id}/report`;
    $("prov-download").href = `/api/runs/${id}/provenance`;
    $("provn-download").href = `/api/runs/${id}/provenance?format=provn`;
    json("provenance-json", provenance);
    $("run-panel").hidden = false;
    $("copy-markdown").disabled = $("download-markdown").disabled = $("copy-provenance").disabled = false;
    $("report-status").textContent = "";
  } catch (error) {
    if (sequence === reportSequence) $("report-status").textContent = `Reports unavailable: ${error.message}`;
    throw error;
  }
}

function downloadMarkdown(report, filename) {
  const url = URL.createObjectURL(new Blob([report], {type: "text/markdown;charset=utf-8"}));
  const link = document.createElement("a"); link.href = url; link.download = filename;
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$("download-markdown").addEventListener("click", () => {
  if (currentRunMarkdown) downloadMarkdown(currentRunMarkdown, `run-${currentRun.id}.md`);
});
$("copy-provenance").addEventListener("click", async () => {
  if (!currentProvenance) return;
  try {
    await copyToClipboard(JSON.stringify(currentProvenance, null, 2), "Provenance JSON");
    $("provenance-copy-status").textContent = "";
  } catch { $("provenance-copy-status").textContent = "Clipboard unavailable. Use Download PROV-JSON."; }
});

$("copy-markdown").addEventListener("click", async () => {
  const report = currentRunMarkdown;
  if (!report) return;
  try {
    await copyToClipboard(report, "Run Markdown");
    $("markdown-copy-status").textContent = "";
  } catch {
    $("markdown-copy-status").textContent = "Clipboard unavailable. Use Download report to save the Markdown.";
  }
});

$("question-form").addEventListener("submit", guarded(async () => {
  if (runningExperiment) return;
  if (!editingExperiment) throw new Error("Select a saved experiment first.");
  if (!contextPreview || previewKey !== contextConfigurationKey()) throw new Error("Wait for the context preview before running.");
  const prepared = {system_prompt: $("run-system-prompt").value, llm_context: $("run-context").value, context_chunk_ids: contextPreview.chunk_ids};
  const total = Number($("repetitions").value);
  const {documents: ownedDocumentConfiguration, ...executionSelection} = researchSelection();
  const definition = definitionFromEditor();
  if (definition.output_mode === "facts_and_answer") definition.semantic_evaluator = executionEvaluatorConfig();
  const payload = {definition, saved_id: editingExperiment,
    execution: {repetitions: total, ...executionSelection, ...serviceSelection(), ...prepared, ...promptSelection()}};
  runningExperiment = true;
  $("run-button").disabled = true;
  try {
    const execution = await api("/executions", payload);
    await followExecution(execution);
  } finally { runningExperiment = false; updateExperimentState(); }
}));

$("sync-files").addEventListener("click", () => {
  $("file-sync-status").textContent = "";
  $("file-sync-dialog").showModal();
});
$("close-file-sync").addEventListener("click", () => $("file-sync-dialog").close());
for (const direction of ["import", "export"]) {
  $(`${direction}-library-files`).addEventListener("click", async () => {
    const message = direction === "import"
      ? "Replace ALL saved experiments and prompts with the JSON files on disk? Unsaved edits will be discarded. Recorded runs and servers are kept."
      : "Replace ALL JSON files in experiments/ and prompts/ with your saved experiments and prompts? Unsaved edits are not exported.";
    if (!window.confirm(message)) return;
    for (const id of ["import-library-files", "export-library-files", "close-file-sync"]) $(id).disabled = true;
    $("file-sync-status").textContent = "Syncing…";
    try {
      const result = await api(`/library-files/${direction}`, {});
      if (direction === "import") {
        clearExperimentFields(); setExperimentEditing(false);
        await refreshDefinitions();
        await refreshPrompts();
        refreshContextDraft();
      }
      $("file-sync-status").textContent = `${direction === "import" ? "Loaded" : "Exported"} ${result.experiments} experiments and ${result.prompts} prompt sets.`;
    } catch (error) { $("file-sync-status").textContent = error.message; }
    finally {
      for (const id of ["import-library-files", "export-library-files", "close-file-sync"]) $(id).disabled = false;
    }
  });
}

let activeExecutionId = null, wakeExecution = null, executionUpdate = null;
async function reconnectExecution() {
  let execution = await api("/executions/active");
  const previous = localStorage.getItem("active-execution");
  if (!execution && previous) {
    try { execution = await api(`/executions/${previous}`); }
    catch (error) { if (error.status === 404) localStorage.removeItem("active-execution"); else throw error; }
  }
  if (!execution) return;
  if (execution.saved_id && experimentDefinitions.some(item => item.id === execution.saved_id)) {
    $("saved-experiment").value = execution.saved_id;
    $("saved-experiment").dispatchEvent(new Event("change"));
  }
  const config = execution.execution || {};
  for (const [field, id] of [["answer_prompt_id", "answer-prompt"], ["review_facts_prompt_id", "review-facts-prompt"], ["review_answer_prompt_id", "review-answer-prompt"]]) {
    if (config[field] && [...$(id).options].some(item => item.value === config[field])) $(id).value = config[field];
  }
  if (serviceProfiles.some(profile => profile.id === config.llm_profile_id)) $("llm-profile").value = config.llm_profile_id;
  if (config.temperature != null) { $("temperature").value = config.temperature; showTemperature("temperature"); }
  $("repetitions").value = execution.total;
  if (config.max_tokens != null) $("max-tokens").value = config.max_tokens;
  $("seed-enabled").checked = config.seed != null;
  if (config.seed != null) $("seed").value = config.seed;
  $("seed-enabled").dispatchEvent(new Event("change"));
  setEvaluatorConfig(execution.reviewer, false);
  await followExecution(execution);
}

async function followExecution(execution) {
  runningExperiment = true; executionUpdate = null; activeExecutionId = execution.batch_id;
  localStorage.setItem("active-execution", activeExecutionId);
  const controls = [...document.querySelectorAll("#experiment-settings input, #experiment-settings select, #experiment-settings textarea, #experiment-settings button, #experiments-manager input, #experiments-manager select, #experiments-manager textarea, #experiments-manager button, #settings-menu input, #settings-menu select, #settings-menu button")]
    .filter(node => node.id !== "cancel-execution").map(node => [node, node.disabled]);
  for (const [node] of controls) node.disabled = true;
  $("batch-results").hidden = true; $("run-panel").hidden = true; $("run-reports").open = false; ++reportSequence; currentRunMarkdown = "";
  $("execution-progress").hidden = false; $("execution-spinner").hidden = false;
  $("cancel-execution").hidden = false;
  let renderedResults = "";
  try {
    while (true) {
      if (executionUpdate) { execution = executionUpdate; executionUpdate = null; }
      const active = ["running", "cancelling"].includes(execution.status);
      $("execution-counter").max = execution.total;
      $("execution-counter").value = execution.completed;
      $("cancel-execution").disabled = execution.status !== "running";
      $("progress").textContent = `${execution.title} · ${execution.completed} / ${execution.total} runs completed · ${execution.status === "cancelling" ? "stopping the active run…" : execution.status}`;
      const resultsKey = `${execution.completed}:${execution.status}`;
      if (execution.summary && resultsKey !== renderedResults) {
        renderBatchResults(execution.summary);
        renderedResults = resultsKey;
      }
      if (!active) {
        if (execution.error) $("progress").textContent += ` · ${execution.error}`;
        localStorage.removeItem("active-execution");
        break;
      }
      await new Promise(resolve => {
        const done = () => { clearTimeout(timer); wakeExecution = null; resolve(); };
        const timer = setTimeout(done, 1000); wakeExecution = done;
      });
      try { execution = await api(`/executions/${activeExecutionId}?known_completed=${execution.completed}`); }
      catch (error) {
        if (error.status === 404) {
          localStorage.removeItem("active-execution");
          throw error;
        }
        $("progress").textContent = "Connection lost · reconnecting to the running experiment…";
      }
    }
  } finally {
    activeExecutionId = null; wakeExecution = null; executionUpdate = null; runningExperiment = false;
    $("execution-spinner").hidden = true; $("cancel-execution").hidden = true;
    for (const [node, disabled] of controls) node.disabled = disabled;
    updateExperimentState();
  }
}

$("cancel-execution").addEventListener("click", guarded(async () => {
  if (!activeExecutionId) return;
  $("cancel-execution").disabled = true;
  const identity = activeExecutionId;
  try {
    const updated = await api(`/executions/${identity}/cancel`, {});
    if (activeExecutionId !== identity) return;
    executionUpdate = updated;
    $("progress").textContent = "Stopping the active run · keeping completed results…";
    wakeExecution?.();
  } catch (error) { $("cancel-execution").disabled = false; throw error; }
}));

let exportBatch = null;
function renderBatchResults(batch) {
  const expanded = new Map([...$("property-chart").children].map(row => [
    row.dataset.property, [...row.querySelectorAll("details")].map(detail => detail.open)
  ]));
  const selectedRun = $("batch-run-selection").value;
  exportBatch = batch;
  renderExperimentOverview(batch.overview);
  $("batch-results").hidden = false; $("property-chart").replaceChildren();
  $("batch-results-title").textContent = batch.experiment_title ? `Results · ${batch.experiment_title}` : "Experiment results";
  for (const property of batch.property_counts || []) {
    const row = element("div", undefined, "property-bar"), heading = element("div", undefined, "property-bar-heading");
    row.dataset.property = property.id;
    heading.append(element("strong", property.id === "answer_json" ? "Answer JSON · format" : property.id === "facts" ? "Facts · content and sources" : property.id === "answer" ? "Answer" : property.id), element("span", `${property.pass} / ${property.total}`));
    const results = propertyRunResults(batch, property.id);
    const cautionCount = results.rows.filter(item => item.caution).length;
    const caution = element("div", undefined, "property-bar-caution");
    caution.style.width = `${property.total ? 100 * cautionCount / property.total : 0}%`;
    caution.title = `${cautionCount} passed with reviewer confidence below 100`;
    const track = element("div", undefined, "property-bar-track"), fill = element("div", undefined, "property-bar-fill");
    const failed = element("div", undefined, "property-bar-fail");
    const review = element("div", undefined, "property-bar-review");
    track.setAttribute("role", "meter"); track.setAttribute("aria-label", property.id);
    track.setAttribute("aria-valuemin", "0"); track.setAttribute("aria-valuemax", String(property.total));
    track.setAttribute("aria-valuenow", String(property.pass));
    track.setAttribute("aria-valuetext", `${property.pass} passed (${cautionCount} with confidence below 100), ${property.fail} failed, ${property.requires_review} require review, ${property.not_evaluated} not evaluated out of ${property.total} attempts`);
    fill.style.width = `${property.total ? 100 * (property.pass - cautionCount) / property.total : 0}%`;
    failed.style.width = `${property.total ? 100 * property.fail / property.total : 0}%`;
    fill.title = `${property.pass - cautionCount} passed without confidence warning`; failed.title = `${property.fail} failed`;
    review.style.width = `${property.total ? 100 * property.requires_review / property.total : 0}%`;
    review.title = `${property.requires_review} require review; correctness has not been established`;
    track.append(fill, caution, failed, review);
    if (property.id === "facts" || property.id === "answer") {
      track.replaceChildren();
      track.classList.add("correctness-track");
      track.setAttribute("role", "group");
      track.setAttribute("aria-label", `${property.id} · reviewer correctness (confidence)`);
      for (const name of ["aria-valuemin", "aria-valuemax", "aria-valuenow", "aria-valuetext"]) track.removeAttribute(name);
      for (const [index, segments] of correctnessSegments(results.rows).entries()) {
        const run = element("div", undefined, "correctness-run");
        const summary = correctnessSummary(results.rows[index].reviewerScores);
        for (const score of segments) {
          const segment = element("span", undefined, "correctness-segment");
          if (score.color || score.citationFailed) segment.style.backgroundColor = score.citationFailed ? "#cf4b4b" : score.color;
          if (score.citationFailed) segment.classList.add("citation-invalid-segment");
          segment.title = `${summary.tooltip}\n\n${score.tooltip}`;
          segment.setAttribute("aria-label", score.tooltip);
          segment.tabIndex = 0;
          run.append(segment);
        }
        run.setAttribute("aria-label", `Run ${results.rows[index].number} · ${summary.tooltip}`);
        run.title = summary.tooltip;
        if (results.rows.length <= 5) {
          const label = element("span", summary.text, "correctness-summary");
          label.setAttribute("aria-hidden", "true");
          run.append(label);
        }
        track.append(run);
      }
    }

    row.append(heading, track, element("p", `${property.fail} failed checks · ${property.requires_review} require review · ${property.not_evaluated} not evaluated${property.evaluation_error ? ` · ${property.evaluation_error} evaluation errors` : ""}`, "muted"));
    const responses = element("div", undefined, "property-responses");
    responses.dataset.property = property.id;
    const foldExpected = results.factsMode && (results.format || ["facts", "answer"].includes(property.id));
    const expected = foldExpected ? element("details", undefined, "property-expected-details") : responses;
    const expectedHeading = element(foldExpected ? "summary" : "h4", results.factsMode ? (results.format ? "Required answer structure" : property.id === "facts" ? "Expected facts and sources" : property.id === "answer" ? "Expected answer" : "Source identifier rule") : results.format ? "Expected output · Validation JSON" : "Expected value · Validation JSON");
    if (results.factsMode && ["answer", "facts"].includes(property.id)) {
      const source = element("button", "</>", "secondary reviewer-prompt-button");
      source.type = "button"; source.title = property.id === "facts" ? "View fact reviewer prompt" : "View reviewer prompt";
      source.setAttribute("aria-label", source.title);
      source.setAttribute("aria-haspopup", "dialog");
      source.setAttribute("aria-controls", "reviewer-prompt-dialog");
      source.addEventListener("click", event => { event.preventDefault(); openReviewerPrompt(batch, property.id === "facts" ? "fact" : "answer"); });
      expectedHeading.append(source);
    }
    expected.append(expectedHeading);
    expected.append(element("pre", results.expectationsVary ? "Expectations differ between runs; shown individually below."
      : results.hasExpected ? JSON.stringify(results.expected, null, 2) : "No expected value recorded.", "property-expected"));
    if (foldExpected) responses.append(expected);
    if (results.format) responses.append(element("p", results.factsMode ? "Checks structure and types only. Facts, sources and answer are checked separately below." : "This bar checks JSON structure only. Expected field values are checked separately below.", "muted"));
    if (results.factsMode && !results.format) responses.append(element("p", "Meaning and evidence support use recorded evaluator judgments. Facts also require exact expected sources. This is not a deterministic proof of correctness.", "muted"));
    const answers = element("details", undefined, "property-run-details");
    answers.append(element("summary", `Responses per run (${results.rows.length})`));
    const list = element("ol", undefined, "property-run-list");
    list.setAttribute("aria-label", `Responses per run for ${property.id}`);
    for (const item of results.rows) {
      const state = item.caution ? "caution" : item.result === "pass" ? "pass" : item.result === "fail" ? "fail" : "review";
      const entry = element("li", undefined, `property-result-${state}`);
      const labels = {pass: "Pass", fail: "Fail", requires_review: "Requires review", not_evaluated: "Not evaluated", evaluation_error: "Evaluation error"};
      const line = element("div", undefined, "property-run-heading");
      const label = element("span", labels[item.result] || item.result, "property-run-status muted");
      label.title = item.reason;
      if (item.caution) label.textContent = "Pass · below 100 confidence";
      if (item.jsonRepairStatus) label.textContent += item.jsonRepairStatus === "repaired" ? " · JSON repaired" : " · JSON repair unsuccessful";
      if (item.method === "exact_reference") label.textContent += " · exact reference match";
      if (item.confidence != null) label.textContent += ` · confidence ${item.confidence}/100`;
      if (item.confidenceThreshold != null) label.textContent += ` · minimum ${item.confidenceThreshold}`;
      let value;
      if (item.sourceCheck && item.result === "pass") value = "All cited source identifiers match supplied sources. This does not establish factual support.";
      else if (item.hasObserved) value = JSON.stringify(item.observed, null, 2);
      else {
        const raw = item.rawAnswer ?? (item.error ? `Run failed: ${item.error}` : "No response recorded.");
        value = results.format || item.invalidFormat ? raw : `${item.missing ? "Field missing or response not assessable." : "No assessed field value recorded."} Raw response:\n${raw}`;
      }
      line.append(element("strong", `Run ${item.number}`), label);
      let readable = value;
      try { readable = JSON.stringify(JSON.parse(value), null, 2); } catch { /* Show non-JSON output unchanged. */ }
      const answer = element("div", undefined, "property-answer");
      answer.append(element("strong", results.format || item.invalidFormat ? "Answer model response" : "Returned value"), element("pre", readable, "property-response-value"));
      entry.append(line, answer);
      if (results.factsMode && !results.format) {
        if (!item.reviewerScores.length) entry.append(element("p", "Reviewer scores: unavailable · no model judgment recorded.", "muted"));
        else {
          const scores = element("table", undefined, "reviewer-scores");
          const header = element("tr");
          for (const [name, help] of [["Statement", "Each generated fact and the answer are judged separately."],
            ["Correctness", "Semantic correctness relative to the reference, independent of citation validity. Observational only; not a calibrated probability."],
            ["Confidence", "How certain the reviewer is about its judgment, whether positive or negative. The existing review threshold applies to confidence."],
            ["Semantic meaning", "Semantic equivalence to the expected statement."],
            ["Source support", "Whether the actual cited document supports the generated statement."],
            ["Citation validation", "Strict deterministic quotation matching. High semantic scores do not override a failed citation."]]) {
            const cell = element("th", name); cell.title = help; header.append(cell);
          }
          const head = element("thead"); head.append(header); scores.append(head);
          const body = element("tbody");
          for (const score of item.reviewerScores) {
            const line = element("tr"); line.title = `${score.meaning || "Unavailable"} · ${score.support || "Unavailable"}. ${score.explanation}`;
            for (const value of [score.label, score.correctnessScore == null ? "Unavailable" : `${score.correctnessScore}/100`,
              score.confidence == null ? "Unavailable" : `${score.confidence}/100`, score.meaning || "Unavailable", score.support || "Unavailable",
              score.property === "facts" ? citationStatus(score.citations) : "Not applicable"]) line.append(element("td", value));
            if (score.citations.some(check => check.status === "fail")) line.classList.add("citation-failed");
            body.append(line);
          }
          scores.append(body); entry.append(scores);
        }
        if (property.id === "facts") {
          for (const check of item.citationChecks) {
            const diagnostic = element("div", undefined, check.status === "fail" ? "citation-failed citation-diagnostic" : "citation-diagnostic");
            diagnostic.append(element("strong", `Fact ${check.fact_index + 1} · ${check.document_id} v${check.version} · Citation ${citationStatus([check])}`));
            for (const error of check.errors || []) {
              diagnostic.append(element("p", error.code || error.reason));
              diagnostic.append(element("pre", JSON.stringify({first_difference: error.first_difference, quote_excerpt: error.quote_excerpt, source_excerpt: error.source_excerpt}, null, 2)));
            }
            entry.append(diagnostic);
          }
        }
      }
      if (["fail", "not_evaluated", "evaluation_error"].includes(item.result)) {
        const reason = element("p", undefined, "property-result-reason");
        reason.append(element("strong", item.result === "fail" ? "Why this failed: " : "Reason: "), document.createTextNode(item.displayReason));
        entry.append(reason);
      }
      if (results.expectationsVary) entry.append(element("p", "Expected"), element("pre", item.hasExpected ? JSON.stringify(item.expected, null, 2) : "Not recorded"));
      list.append(entry);
    }
    answers.append(list); responses.append(answers); row.append(responses);
    row.querySelectorAll("details").forEach((detail, index) => {
      detail.open = expanded.get(property.id)?.[index] ?? false;
    });
    $("property-chart").append(row);
  }
  if (!$("property-chart").children.length) $("property-chart").append(element("p", "No saved automatic checks are available for this batch."));
  $("batch-run-selection").replaceChildren();
  batch.runs.forEach((run, index) => option($("batch-run-selection"), run.id, `Run ${index + 1} · ${run.status} · ${run.id.slice(0, 8)}`));
  if (batch.runs.some(run => run.id === selectedRun)) $("batch-run-selection").value = selectedRun;
  else if (currentRun && batch.runs.some((run) => run.id === currentRun.id)) $("batch-run-selection").value = currentRun.id;
}
function renderExperimentOverview(overview) {
  const target = $("experiment-overview"); target.replaceChildren();
  target.append(element("h3", "Experiment Overview"));
  if (!overview) { target.append(element("p", "N/A · performance data not recorded or backend update required.", "muted")); return; }
  const amount = (value) => value == null ? "N/A" : new Intl.NumberFormat("en", {maximumFractionDigits: 1}).format(value);
  const duration = (value) => value == null ? "N/A" : value >= 1000 ? `${new Intl.NumberFormat("en", {maximumFractionDigits: 3}).format(value / 1000)} s` : `${amount(value)} ms`;
  const tokens = overview.generator.tokens, config = overview.configuration;
  const cards = element("div", undefined, "performance-cards");
  const fastest = overview.fastest, slowest = overview.slowest;
  for (const [name, value] of [["Total batch duration", duration(overview.total_duration_ms)], ["Average generator duration", duration(overview.average_generator_ms)],
    ["Fastest generator", fastest ? `Run ${fastest.number} · ${duration(fastest.generator_duration_ms)}` : "N/A"],
    ["Slowest generator", slowest ? `Run ${slowest.number} · ${duration(slowest.generator_duration_ms)}` : "N/A"],
    ["Completed requests", amount(overview.completed_runs)], ["Execution errors", amount(overview.failed_runs)]]) {
    const card = element("div", undefined, "performance-card"); card.append(element("span", name, "muted"), element("strong", value)); cards.append(card);
  }
  target.append(cards, element("p", "Execution errors count failed answer-model requests, such as timeouts. Rejected answers are counted in the checks below.", "muted"), element("p", `Generator duration includes initial generation and any JSON repair (network and prompt processing included). Fastest, slowest and average use completed non-demo generators only: ${overview.duration_measured_runs}/${overview.completed_generator_runs} timed. Batch duration includes setup and evaluation; ${overview.batch_duration_basis}.`, "muted"));
  const format = overview.generator.json_format;
  if (format?.checked) target.append(element("p", `Initial JSON format valid: ${format.initial_valid}/${format.checked} · JSON repairs successful: ${format.repaired}/${format.repair_attempts}. Repair calls are included in generator time and token totals.`, "muted"));
  const throughput = (value) => {
    if (value?.decode_tokens_per_second != null) return `${amount(value.decode_tokens_per_second)} tokens/s · provider decode (${value.decode_measured_calls}/${value.calls} calls)`;
    if (value?.end_to_end_tokens_per_second != null) return `${amount(value.end_to_end_tokens_per_second)} tokens/s · end-to-end throughput (${value.end_to_end_measured_calls}/${value.calls} calls)`;
    return "N/A";
  };
  const table = element("table"); table.className = "performance-table";
  const head = element("tr"); for (const label of ["Generator", "Request duration", "Output tokens", "Speed"]) head.append(element("th", label));
  const thead = element("thead"); thead.append(head); table.append(thead);
  const body = element("tbody");
  const completed = overview.runs.filter((run) => run.status === "completed" && run.backend !== "demo");
  const outputs = completed.map((run) => run.generator_tokens.output_tokens);
  for (const [name, run] of [["Fastest", fastest], ["Average", null], ["Slowest", slowest]]) {
    const row = element("tr"), label = element("td");
    if (run) {
      const link = element("button", `${name} · Run ${run.number}`, "secondary"); link.type = "button";
      link.addEventListener("click", guarded(async () => { await loadRun(run.run_id); $("run-reports").open = true; $("run-reports").scrollIntoView({block: "start"}); })); label.append(link);
    } else label.textContent = name;
    row.append(label, element("td", duration(run ? run.generator_duration_ms : name === "Average" ? overview.average_generator_ms : null)),
      element("td", amount(run ? run.generator_tokens.output_tokens : name === "Average" && outputs.length && outputs.every((n) => n != null) ? outputs.reduce((a,b) => a+b,0) / outputs.length : null)),
      element("td", throughput(run ? run.generator_speed : name === "Average" ? overview.generator : null))); body.append(row);
  }
  table.append(body); const tableWrap = element("div", undefined, "performance-scroll"); tableWrap.append(table); target.append(tableWrap);
  const chart = element("details"); chart.append(element("summary", "Generator duration per run"));
  const maxDuration = Math.max(0, ...overview.runs.map((run) => run.generator_duration_ms ?? 0));
  for (const run of overview.runs) {
    const row = element("div", undefined, "performance-duration-row"); row.append(element("span", `Run ${run.number} · ${run.status}`));
    const track = element("div", undefined, "property-bar-track"), fill = element("div", undefined, run.status === "failed" ? "property-bar-fail" : "property-bar-fill");
    fill.style.width = `${maxDuration ? 100 * (run.generator_duration_ms ?? 0) / maxDuration : 0}%`; track.append(fill);
    track.setAttribute("aria-label", `Run ${run.number}: ${duration(run.generator_duration_ms)}`);
    row.append(track, element("span", duration(run.generator_duration_ms))); chart.append(row);
  }
  target.append(chart, element("h4", "Generator tokens and speed"), element("p", "Token totals cover all recorded answer-model calls in this batch. Review-model tokens are counted separately.", "muted"));
  const usage = element("dl", undefined, "performance-values");
  const tokenValue = (key) => `${amount(tokens[key])} · ${tokens.reported[key]}/${tokens.calls} calls reported`;
  for (const [label, value] of [["Total input tokens", tokenValue("input_tokens")], ["Total output tokens", tokenValue("output_tokens")], ["Total tokens (input + output)", tokenValue("total_tokens")],
    ["Generation speed", throughput(overview.generator)], ["Average time to first token", `${duration(overview.generator.average_ttft_ms)} · provider-reported (${overview.generator.ttft_measured_calls} calls)`]]) usage.append(element("dt", label), element("dd", value));
  if (tokens.reported.cached_input_tokens) usage.append(element("dt", "Total cached input tokens"), element("dd", `${tokenValue("cached_input_tokens")} · subset of input, never added again`));
  target.append(usage);
  target.append(element("h4", "Recorded configuration"));
  const modelName = (model) => model?.name || "Unknown model";
  const spread = (value, unit) => value.measured_runs ? `${amount(value.min)}${value.min !== value.max ? `–${amount(value.max)}` : ""} ${unit}/run` : "N/A";
  const values = element("dl", undefined, "performance-values");
  for (const [label,value] of [["Generator model", config.models.map(modelName).join("; ") || "N/A"], ["Temperature", config.temperatures.map(amount).join("; ") || "N/A"],
    ["Seed", config.seeds.map(amount).join("; ") || "N/A"], ["Unique enabled document versions", amount(config.unique_enabled_document_versions)],
    ["Supplied source fragments", spread(config.source_fragments,"fragments")], ["Supplied source text", spread(config.source_text_characters,"characters")], ["Source-only tokens", "N/A · not separately measured; prompt tokens also include instructions and question"]]) values.append(element("dt",label),element("dd",value));
  target.append(values);
  const evaluator = overview.evaluator;
  if (evaluator.configured) {
    const detail = element("details", undefined, "performance-evaluator"); detail.append(element("summary", `Evaluator overhead · ${evaluator.calls} calls · ${evaluator.errors} errors`));
    const values = element("dl", undefined, "performance-values");
    for (const [label,value] of [["Evaluator model",evaluator.models.map(modelName).join("; ") || "N/A"], ["Evaluator calls",amount(evaluator.calls)], ["Total evaluator time",duration(evaluator.total_duration_ms)],
      ["Average evaluator time",`${duration(evaluator.average_duration_ms)} · ${evaluator.duration_measured_calls}/${evaluator.calls} calls timed`], ["Total evaluator input tokens",amount(evaluator.tokens.input_tokens)], ["Total evaluator output tokens",amount(evaluator.tokens.output_tokens)], ["Evaluator errors",amount(evaluator.errors)]]) values.append(element("dt",label),element("dd",value));
    detail.append(values); target.append(detail);
  }
}
$("run-reports").addEventListener("toggle", guarded(async () => {
  const id = $("batch-run-selection").value;
  if ($("run-reports").open && id && (currentRun?.id !== id || !currentRunMarkdown)) await loadRun(id);
}));
$("batch-run-selection").addEventListener("change", guarded(async () => {
  if ($("batch-run-selection").value) await loadRun($("batch-run-selection").value);
}));
guarded(async () => {
  serviceProfiles = await loadLLMProfiles();
  renderProfiles(true);
  if (!serviceProfiles.length) openServerManager();
  await installPromptManager(() => { previewKey = ""; scheduleContextPreview(); updateExperimentState(); });
  await refreshDefinitions();
  refreshResearchConfiguration();
  await reconnectExecution();
})();

function jsonObject(id) {
  if (id === "validation-json" && !$(id).value.trim()) return {};
  let value;
  try { value = JSON.parse($(id).value); } catch { throw new Error(`${id === "answer-json" ? "Answer" : "Validation"} JSON must be valid JSON.`); }
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Use a JSON object.");
  return value;
}
function setEvaluatorConfig(config, migrate = true) {
  const migrated = localStorage.getItem("review-threshold-default-90") === "done";
  config = reviewSettings(config, migrated || !migrate);
  if (!migrated && migrate) {
    localStorage.setItem("review-settings", JSON.stringify(config));
    localStorage.setItem("review-threshold-default-90", "done");
  }
  const id = serviceProfiles.some(profile => profile.id === config?.llm_profile_id)
    ? config.llm_profile_id : (serviceProfiles[0]?.id || "");
  $("semantic-evaluator").value = id;
  $("evaluator-temperature").value = config?.temperature ?? 0;
  showTemperature("evaluator-temperature");
  $("evaluator-seed").value = config?.seed ?? 42;
  $("evaluator-max-tokens").value = !migrate && config?.max_tokens != null ? config.max_tokens : "";
  $("evaluator-threshold").value = config?.confidence_threshold ?? 90;
}
function updateOutputMode() {
  const facts = $("output-mode").value === "facts_and_answer";
  $("semantic-options").hidden = !facts;
  $("answer-json").readOnly = facts || !managerEditing;
  $("validation-json").readOnly = facts || !managerEditing;
  $("validation-json").hidden = facts;
  $("validation-help").hidden = facts;
  document.querySelector('label[for="validation-json"]').hidden = facts;
  if (facts) $("answer-json").value = JSON.stringify(factAnswerTemplate, null, 2);
}
let savedEvaluatorConfig = null;
$("llm-profile").addEventListener("change", () => {
  localStorage.setItem("answer-model", $("llm-profile").value);
});
function executionEvaluatorConfig() {
  return $("semantic-evaluator").value ? {
    llm_profile_id: $("semantic-evaluator").value, temperature: temperatureValue("evaluator-temperature"),
    confidence_threshold: Number($("evaluator-threshold").value),
    seed: $("seed-enabled").checked ? Number($("seed").value) : null, max_tokens: outputTokenLimit($("evaluator-max-tokens").value, serviceProfiles.find((item) => item.id === $("semantic-evaluator").value), 4096)
  } : null;
}
for (const id of ["semantic-evaluator", "evaluator-temperature", "evaluator-seed", "evaluator-max-tokens", "evaluator-threshold"]) {
  $(id).addEventListener("change", () => {
    const config = executionEvaluatorConfig();
    if (config) delete config.max_tokens;
    localStorage.setItem("review-settings", JSON.stringify(config));
    localStorage.setItem("review-threshold-default-90", "done");
    updateExperimentState();
  });
}
let experimentJSONActive = false;
function setExperimentJSONView(active) {
  experimentJSONActive = active;
  $("experiment-form-fields").hidden = active;
  $("experiment-json-pane").hidden = !active;
  $("experiment-form-view").setAttribute("aria-pressed", String(!active));
  $("experiment-json-view").setAttribute("aria-pressed", String(active));
  $("experiment-json-error").hidden = true;
}
async function applyExperimentJSON() {
  const definition = parseExperimentJSON($("experiment-json").value);
  const draft = await api("/experiment-definitions/import", {format: "traceability-experiment-v2", experiment: definition});
  populateExperimentDefinition(draft.definition);
  setExperimentEditing(managerEditing);
  updateExperimentState();
}
$("experiment-json-view").addEventListener("click", guarded(() => {
  if (experimentJSONActive) return;
  $("experiment-json").value = JSON.stringify(experimentJSONValue(definitionFromEditor()), null, 2);
  setExperimentJSONView(true);
}));
$("experiment-form-view").addEventListener("click", async () => {
  if (!experimentJSONActive) return;
  try { await applyExperimentJSON(); }
  catch (error) { $("experiment-json-error").textContent = error.message; $("experiment-json-error").hidden = false; }
});
$("experiment-json").addEventListener("input", () => {
  $("experiment-json-error").hidden = true;
  try { parseExperimentJSON($("experiment-json").value); }
  catch (error) { $("experiment-json-error").textContent = error.message; $("experiment-json-error").hidden = false; }
  updateExperimentState();
});
function definitionFromEditor() {
  if (experimentJSONActive) return parseExperimentJSON($("experiment-json").value);
  if (missingDefinitionSources.length) throw new Error("A saved source is unavailable. Explicitly update the source selection before saving or running.");
  const factsMode = $("output-mode").value === "facts_and_answer";
  let reference = null;
  if (factsMode && $("semantic-reference").value.trim()) {
    try { reference = JSON.parse($("semantic-reference").value); } catch { throw new Error("Expected facts and answer must be valid JSON."); }
    if (!reference || typeof reference !== "object" || Array.isArray(reference)) throw new Error("Use an object for the semantic reference.");
  }
  return {output_mode: $("output-mode").value, semantic_reference: reference,
    semantic_evaluator: factsMode ? savedEvaluatorConfig : null,
    title: $("experiment-title").value, question: $("question").value,
    version_ids: [], documents: experimentDocuments,
    answer_json: factsMode ? factAnswerTemplate : jsonObject("answer-json"), validation_json: factsMode ? {} : jsonObject("validation-json")};
}
async function refreshDefinitions() {
  experimentDefinitions = (await api("/experiment-definitions")).map((item) => item.definition.documents == null
    ? {...item, definition: {...item.definition, documents: item.document_copies || [], version_ids: []}}
    : item);
  $("saved-experiment").replaceChildren(); option($("saved-experiment"), "", "Select an experiment");
  for (const item of experimentDefinitions) option($("saved-experiment"), item.id, item.definition.title);
  $("saved-experiment").value = editingExperiment || "";
  renderExperimentManager();
}
$("saved-experiment").addEventListener("change", () => {
  generatedValidation = null;
  $("batch-results").hidden = true; $("run-panel").hidden = true; $("run-reports").open = false; ++reportSequence; currentRunMarkdown = "";
  const item = experimentDefinitions.find((entry) => entry.id === $("saved-experiment").value);
  if (!item) { clearExperimentFields(); setExperimentEditing(false); return; }
  refreshResearchConfiguration();
  editingExperiment = item.id;
  const missing = populateExperimentDefinition(item.definition);
  $("definition-status").textContent = missing.length ? "A saved source is unavailable. Select its replacement before saving." : "";
  $("managed-experiment").value = item.id;
});
function populateExperimentDefinition(input) {
  setExperimentJSONView(false);
  const definition = definitionDefaults(input);
  $("legacy-prompt-warning").hidden = !(input.system_prompt || input.context);
  $("output-mode").value = definition.output_mode;
  $("semantic-reference").value = definition.semantic_reference ? JSON.stringify(definition.semantic_reference, null, 2) : "";
  savedEvaluatorConfig = definition.semantic_evaluator;
  if (!localStorage.getItem("review-settings")) setEvaluatorConfig(savedEvaluatorConfig);
  $("experiment-title").value = definition.title; $("question").value = definition.question;
  $("answer-json").value = JSON.stringify(definition.answer_json, null, 2);
  $("validation-json").value = JSON.stringify(definition.validation_json, null, 2);
  updateOutputMode();
  experimentDocuments = structuredClone(definition.documents || []);
  renderOwnedDocuments();
  const missing = [];
  missingDefinitionSources = [];
  refreshContextDraft();
  return missing;
}
let missingDefinitionSources = [];
let managerEditing = false, managerRestoreId = null;
function clearExperimentFields() {
  $("legacy-prompt-warning").hidden = true;
  setExperimentJSONView(false);
  generatedValidation = null; experimentDocuments = []; renderOwnedDocuments();
  editingExperiment = null; missingDefinitionSources = []; $("saved-experiment").value = "";
  $("output-mode").value = "facts_and_answer"; $("semantic-reference").value = ""; savedEvaluatorConfig = null;
  $("experiment-title").value = ""; $("question").value = "";
  $("answer-json").value = JSON.stringify(factAnswerTemplate, null, 2); $("validation-json").value = "{}";
  updateOutputMode();
  refreshResearchConfiguration(); refreshContextDraft();
}
function setExperimentEditing(enabled) {
  managerEditing = enabled;
  $("experiment-editor").hidden = !enabled && !editingExperiment;
  $("manager-empty-state").hidden = enabled || Boolean(editingExperiment);
  for (const node of $("experiment-editor").querySelectorAll("input, textarea")) node.readOnly = !enabled;
  for (const select of $("experiment-editor").querySelectorAll("select")) select.disabled = !enabled;
  updateOutputMode();
  $("add-owned-document").disabled = runningExperiment || (!enabled && !editingExperiment);
  $("import-owned-documents").disabled = runningExperiment || (!enabled && !editingExperiment);
  renderOwnedDocuments();
  $("save-experiment").disabled = !enabled;
  $("managed-experiment").disabled = enabled;
  $("new-experiment").disabled = enabled;
  $("edit-experiment").disabled = enabled || !editingExperiment;
  $("export-experiment").disabled = enabled || !editingExperiment;
  $("delete-experiment").disabled = enabled || !editingExperiment;
  $("import-experiment").hidden = !enabled || Boolean(editingExperiment);
  $("experiment-json").readOnly = !enabled;
  $("experiment-form-view").disabled = $("experiment-json-view").disabled = !enabled && !editingExperiment;
}
$("new-experiment").addEventListener("click", () => {
  managerRestoreId = editingExperiment;
  clearExperimentFields(); $("managed-experiment").value = "";
  setExperimentEditing(true);
  $("definition-status").textContent = "New experiment · choose sources and their order.";
  $("experiment-title").focus();
});
$("delete-experiment").addEventListener("click", guarded(async () => {
  if (!editingExperiment || managerEditing || runningExperiment) return;
  const button = $("delete-experiment"); button.disabled = true;
  try {
    await api(`/experiment-definitions/${editingExperiment}`, {}, "DELETE");
    managerRestoreId = null;
    clearExperimentFields();
    $("batch-results").hidden = true; $("run-panel").hidden = true; $("run-reports").open = false; ++reportSequence; currentRunMarkdown = "";
    await refreshDefinitions();
    setExperimentEditing(false);
    $("manager-empty-state").textContent = "Experiment deleted. Recorded runs are preserved. Select an experiment or use + to create one.";
  } finally { button.disabled = managerEditing || !editingExperiment; }
}));
async function saveEditor() {
  const keepJSON = experimentJSONActive;
  const definition = keepJSON
    ? (await api("/experiment-definitions/import", {format: "traceability-experiment-v2", experiment: parseExperimentJSON($("experiment-json").value)})).definition
    : definitionFromEditor();
  const saved = await api(editingExperiment ? `/experiment-definitions/${editingExperiment}` : "/experiment-definitions",
    definition, editingExperiment ? "PUT" : "POST");
  if (keepJSON) populateExperimentDefinition(saved.definition);
  $("legacy-prompt-warning").hidden = !(saved.definition.system_prompt || saved.definition.context);
  editingExperiment = saved.id; missingDefinitionSources = []; await refreshDefinitions();
  updateExperimentState();
  $("definition-status").textContent = "Saved";
  setExperimentEditing(false); managerRestoreId = saved.id;
  if (keepJSON) { $("experiment-json").value = JSON.stringify(experimentJSONValue(definitionFromEditor()), null, 2); setExperimentJSONView(true); }
}
$("save-experiment").addEventListener("click", guarded(async () => {
  $("save-experiment").disabled = true;
  try { await saveEditor(); }
  catch (error) {
    if (!experimentJSONActive) throw error;
    const message = $("experiment-json-error");
    message.textContent = `Cannot save experiment: ${error.message}`;
    message.hidden = false;
    message.scrollIntoView({block: "nearest"});
  }
  finally { $("save-experiment").disabled = !managerEditing; }
}));

function fillEmptyValidation() {
  if ($("output-mode").value === "facts_and_answer") return;
  if (!validationIsEmpty($("validation-json").value) && $("validation-json").value !== generatedValidation) return;
  try {
    const answer = jsonObject("answer-json");
    generatedValidation = JSON.stringify(blankValidation(answer), null, 2);
    $("validation-json").value = generatedValidation;
    updateExperimentState();
  } catch { /* Keep incomplete Answer JSON editable; validation stays untouched. */ }
}
$("answer-json").addEventListener("input", fillEmptyValidation);
$("answer-json").addEventListener("change", fillEmptyValidation);
$("validation-json").addEventListener("input", () => { generatedValidation = null; });
function renderExperimentManager() {
  $("managed-experiment").replaceChildren(); option($("managed-experiment"), "", "Select an experiment");
  for (const item of experimentDefinitions) option($("managed-experiment"), item.id, item.definition.title);
  $("managed-experiment").value = editingExperiment || "";
}
$("managed-experiment").addEventListener("change", () => {
  $("saved-experiment").value = $("managed-experiment").value;
  $("saved-experiment").dispatchEvent(new Event("change"));
  managerRestoreId = editingExperiment; setExperimentEditing(false);
});
$("edit-experiment").addEventListener("click", () => {
  setExperimentEditing(true); $("experiment-title").focus();
});
$("manage-experiments").addEventListener("click", guarded(async () => {
  await refreshDefinitions();
  managerRestoreId = editingExperiment;
  $("saved-experiment").dispatchEvent(new Event("change"));
  setExperimentEditing(false);
  $("settings-menu").close(); $("experiments-manager").showModal();
}));
function cancelExperimentManager() {
  const identity = editingExperiment || managerRestoreId;
  $("saved-experiment").value = identity || "";
  $("saved-experiment").dispatchEvent(new Event("change"));
  setExperimentEditing(false); $("experiments-manager").close();
}
$("cancel-experiment").addEventListener("click", cancelExperimentManager);
$("close-experiments-manager").addEventListener("click", cancelExperimentManager);
$("experiments-manager").addEventListener("cancel", (event) => { event.preventDefault(); cancelExperimentManager(); });
// Native modal handles keyboard focus, Escape and focus restoration on close.
const settingsMenu = $("settings-menu");
function openServerManager() {
  if (!$("edit-profile").value && serviceProfiles.length) $("edit-profile").value = serviceProfiles[0].id;
  profileEditing = false; $("edit-profile").dispatchEvent(new Event("change"));
  if (!settingsMenu.open) settingsMenu.showModal();
  if (!serviceProfiles.length) $("new-profile").click();
  else $("edit-profile").focus();
}
$("manage-servers").addEventListener("click", openServerManager);
$("seed-enabled").addEventListener("change", () => {
  $("seed").disabled = !$("seed-enabled").checked;
  updateExperimentState();
});
$("close-settings").addEventListener("click", () => { if (serviceProfiles.length) settingsMenu.close(); });
settingsMenu.addEventListener("cancel", (event) => { if (!serviceProfiles.length) event.preventDefault(); });
settingsMenu.addEventListener("click", (event) => {
  const bounds = settingsMenu.getBoundingClientRect();
  if (serviceProfiles.length && event.target === settingsMenu && (event.clientX < bounds.left || event.clientX > bounds.right
    || event.clientY < bounds.top || event.clientY > bounds.bottom)) settingsMenu.close();
});

// Reveal advanced controls when native form validation needs focus inside the menu.
$("question-form").addEventListener("invalid", (event) => {
  if (settingsMenu.contains(event.target)) {
    event.target.closest("details")?.setAttribute("open", "");
    if (!settingsMenu.open) settingsMenu.showModal();
  }
}, true);

const dragListeners = new WeakMap();
function resetDragList(list) {
  for (const controller of dragListeners.get(list) || []) controller.abort();
  dragListeners.set(list, []); list.replaceChildren();
}
// Pointer capture supports mouse, pen and touch. Keyboard users reorder with arrows.
function dragHandle(row, list, label, onReorder = () => {}) {
  const handle = element("button", "⠿", "drag-handle secondary");
  handle.type = "button"; handle.setAttribute("aria-label", `Reorder ${label}`);
  const controller = new AbortController(), options = {signal: controller.signal};
  if (!dragListeners.has(list)) dragListeners.set(list, []);
  dragListeners.get(list).push(controller);
  let pointer = null, original = null;
  const commit = () => {
    row.classList.remove("dragging");
    if (original && original !== [...list.children].map((node) => node.dataset.key || node.querySelector("input")?.value).join("|")) {
      Promise.resolve(onReorder()).catch(showError);
    }
    original = null;
  };
  handle.addEventListener("pointerdown", (event) => {
    if (handle.disabled || event.button !== 0) return;
    pointer = event.pointerId;
    original = [...list.children].map((node) => node.dataset.key || node.querySelector("input")?.value).join("|");
    handle.setPointerCapture(pointer); row.classList.add("dragging");
  });
  document.addEventListener("pointermove", (event) => {
    if (event.pointerId !== pointer) return;
    event.preventDefault();
    const siblings = [...list.children].filter((node) => node !== row);
    const next = siblings.find((node) => event.clientY < node.getBoundingClientRect().top + node.getBoundingClientRect().height / 2);
    if (next && row.nextElementSibling !== next) list.insertBefore(row, next);
    else if (!next && list.lastElementChild !== row) list.append(row);
    // Moving a captured element can release capture; restore it after the DOM move.
    if (!handle.hasPointerCapture(pointer)) handle.setPointerCapture(pointer);
  }, options);
  const finish = (event) => {
    if (event.pointerId !== pointer) return;
    pointer = null; commit();
  };
  document.addEventListener("pointerup", finish, options);
  document.addEventListener("pointercancel", finish, options);
  handle.addEventListener("keydown", (event) => {
    if (!["ArrowUp", "ArrowDown"].includes(event.key)) return;
    event.preventDefault();
    const neighbour = event.key === "ArrowUp" ? row.previousElementSibling : row.nextElementSibling;
    if (!neighbour) return;
    event.key === "ArrowUp" ? neighbour.before(row) : neighbour.after(row);
    handle.focus(); Promise.resolve(onReorder()).catch(showError);
  });
  return handle;
}

function contextConfiguration() {
  return {answer_prompt_id: promptSelection().answer_prompt_id, output_mode: $("output-mode").value, ...researchSelection(), ...serviceSelection(), answer_json: jsonObject("answer-json")};
}
function contextConfigurationKey() { return JSON.stringify(contextConfiguration()); }
function scheduleContextPreview() {
  contextPreviewError = "";
  let config;
  try { config = contextConfiguration(); } catch (error) {
    previewKey = ""; contextPreview = null; ++previewSequence; clearTimeout(previewTimer);
    $("run-system-prompt").disabled = true; $("run-context").disabled = true; $("run-button").disabled = true;
    contextPreviewError = error.message;
    $("run-context-status").textContent = error.message;
    return;
  }
  const key = JSON.stringify(config);
  if (key === previewKey) return;
  previewKey = key;
  const sequence = ++previewSequence;
  clearTimeout(previewTimer); contextPreview = null;
  $("run-system-prompt").disabled = true; $("run-context").disabled = true; $("run-button").disabled = true;
  $("run-context-status").textContent = "Preparing the exact LLM context…";
  if (!config.question.trim()) {
    const prompt = selectedAnswerPrompt();
    $("run-system-prompt").value = prompt?.system_prompt || "";
    $("run-context").value = prompt?.user_prompt || "";
    $("run-system-prompt").disabled = $("run-context").disabled = !prompt;
    $("run-context-status").textContent = prompt
      ? "Selected prompt template. Choose an experiment to fill in its question and documents."
      : "Choose a generation prompt to preview it.";
    return;
  }
  previewTimer = setTimeout(async () => {
    try {
      const preview = await api("/context-preview", config);
      if (sequence !== previewSequence) return;
      if (typeof preview.system_prompt !== "string" || typeof preview.text !== "string") throw new Error("Restart the app backend to load the current preview API.");
      contextPreview = preview;
      renderOwnedSourceIds();
      $("run-system-prompt").value = preview.system_prompt; $("run-system-prompt").disabled = false;
      $("run-context").value = preview.text; $("run-context").disabled = false; $("run-button").disabled = false;
      $("run-context-status").textContent = "Exact model input · read only.";
      updateExperimentState();
    } catch (error) {
      if (sequence !== previewSequence) return;
      $("run-context-status").textContent = error.message === "Not Found"
        ? "The running backend does not support this preview. Restart the app backend and reload the page."
        : `Context preview failed: ${error.message}`;
      $("runner-selection-status").textContent = $("run-context-status").textContent;
      previewKey = "";
    }
  }, 300);
}
function refreshContextDraft() { previewKey = ""; scheduleContextPreview(); }
const helpDialog = $("parameter-help");
function showParameterHelp(help) {
    $("parameter-help-title").textContent = help.title;
    $("parameter-help-text").textContent = help.text;
    const options = $("parameter-help-options"); options.replaceChildren();
    options.hidden = !help.entries;
    helpDialog.classList.toggle("expanded-help", Boolean(help.entries));
    for (const entry of help.entries || []) {
      options.append(element("h3", entry.title));
      if (entry.points) {
        const list = element("ul");
        for (const point of entry.points) list.append(element("li", point));
        options.append(list);
      } else {
        for (const [label, text] of [["What we do: ", entry.action], ["What we learn: ", entry.learning]]) {
          const paragraph = element("p"); paragraph.append(element("strong", label), document.createTextNode(text)); options.append(paragraph);
        }
      }
    }
    options.scrollTop = 0;
    helpDialog.showModal();
}
installFieldHelp(document.body, showParameterHelp);
$("close-parameter-help").addEventListener("click", () => helpDialog.close());
helpDialog.addEventListener("click", (event) => {
  const bounds = helpDialog.getBoundingClientRect();
  if (event.target === helpDialog && (event.clientX < bounds.left || event.clientX > bounds.right
    || event.clientY < bounds.top || event.clientY > bounds.bottom)) helpDialog.close();
});

for (const [id, download] of [["copy-batch-markdown", false], ["download-batch-markdown", true]]) {
$(id).addEventListener("click", guarded(async () => {
  if (!exportBatch?.runs?.length) return;
  const buttons = [$("copy-batch-markdown"), $("download-batch-markdown")];
  for (const button of buttons) button.disabled = true;
  try {
    const batch = exportBatch;
    const details = [];
    for (const attempt of batch.runs) {
      const [run, trace] = await Promise.all([api(`/runs/${attempt.id}`), api(`/runs/${attempt.id}/trace`)]);
      details.push({...run, trace});
    }
    const report = failureMarkdown(details[0], batch, details);
    if (download) {
      downloadMarkdown(report, "experiment-analysis.md");
      $("batch-copy-status").textContent = "Analysis Markdown downloaded.";
      return;
    }
    $("batch-markdown-content").value = report;
    try {
      await copyToClipboard(report, "Analysis Markdown");
      $("batch-copy-status").textContent = "";
    } catch {
      $("batch-markdown-fallback").hidden = false;
      $("batch-markdown-content").focus(); $("batch-markdown-content").select();
      $("batch-copy-status").textContent = "Copy the selected Markdown manually.";
    }
  } finally { for (const button of buttons) button.disabled = false; }
}));
}

function downloadJSON(value, filename) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2) + "\n"], {type: "application/json"}));
  const link = document.createElement("a"); link.href = url; link.download = filename;
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$("export-experiment").addEventListener("click", guarded(async () => {
  if (!editingExperiment || managerEditing) return;
  const value = await api(`/experiment-definitions/${editingExperiment}/export`);
  const filename = value.experiment.title.replace(/[^a-zA-Z0-9_-]+/g, "-") || "experiment";
  downloadJSON(value, filename + ".json");
  $("definition-status").textContent = "Experiment exported with its documents.";
}));
$("import-experiment").addEventListener("click", () => openJSONImport("experiment"));
async function importExperimentData(data) {
  const draft = await api("/experiment-definitions/import", data);
  populateExperimentDefinition(draft.definition);
  setExperimentEditing(true); updateExperimentState();
  $("definition-status").textContent = ["Imported draft. Review the fields and Save.", ...draft.warnings].join(" ");
}

const jsonImportDialog = $("json-import-dialog");
let jsonImportKind = null, jsonImportBusy = false, fileDragDepth = 0;
function importError(message) {
  $("json-import-error").textContent = message;
  $("json-import-error").hidden = !message;
}
function clearImportDrag() {
  fileDragDepth = 0; jsonImportDialog.classList.remove("is-file-drag");
}
function openJSONImport(kind) {
  jsonImportKind = kind;
  $("json-import-title").textContent = kind === "experiment" ? "Import experiment" : "Import experiment documents";
  $("json-import-description").textContent = kind === "experiment"
    ? "Paste an experiment JSON object including its documents. Import opens a draft; Save stores it."
    : "Paste a JSON array to add documents to this experiment. Save stores the changes.";
  $("json-import-text").value = ""; $("json-import-filename").textContent = "";
  $("json-import-file").value = ""; importError(""); clearImportDrag();
  jsonImportDialog.showModal(); $("json-import-text").focus();
}
function closeJSONImport() {
  if (jsonImportBusy) return;
  clearImportDrag(); jsonImportDialog.close();
}
$("close-json-import").addEventListener("click", closeJSONImport);
$("cancel-json-import").addEventListener("click", closeJSONImport);
jsonImportDialog.addEventListener("cancel", (event) => { event.preventDefault(); closeJSONImport(); });
jsonImportDialog.addEventListener("close", clearImportDrag);
function setImportBusy(busy) {
  jsonImportBusy = busy;
  $("json-import-text").readOnly = busy;
  for (const id of ["confirm-json-import", "choose-json-import-file", "cancel-json-import", "close-json-import"]) $(id).disabled = busy;
}
async function loadImportFile(files) {
  clearImportDrag();
  if (jsonImportBusy) return;
  if (files.length !== 1) { importError("Drop exactly one JSON file."); return; }
  setImportBusy(true); importError("");
  try {
    $("json-import-text").value = await files[0].text();
    $("json-import-filename").textContent = files[0].name;
  } catch { importError("Could not read the file. Paste the JSON instead."); }
  finally { setImportBusy(false); }
}
$("choose-json-import-file").addEventListener("click", () => $("json-import-file").click());
$("json-import-file").addEventListener("change", async () => {
  if ($("json-import-file").files.length) await loadImportFile([...$("json-import-file").files]);
  $("json-import-file").value = "";
});
$("confirm-json-import").addEventListener("click", async () => {
  if (jsonImportBusy) return;
  importError("");
  let data;
  try {
    data = JSON.parse($("json-import-text").value.replace(/^\uFEFF/, ""));
    if (!data || typeof data !== "object") throw new Error("Object or array required");
  } catch { importError("Enter valid JSON: an experiment object or a documents array."); return; }
  setImportBusy(true);
  try {
    if (jsonImportKind === "experiment") await importExperimentData(data);
    else if (jsonImportKind === "owned-documents") await importOwnedDocumentsData(data);
    else throw new Error("Choose an experiment or experiment-document import.");
    jsonImportDialog.close();
  } catch (error) { importError(error.message); }
  finally { setImportBusy(false); clearImportDrag(); }
});
function draggingFile(event) { return [...(event.dataTransfer?.types || [])].includes("Files"); }
window.addEventListener("dragenter", (event) => {
  if (!jsonImportDialog.open || !draggingFile(event)) return;
  event.preventDefault();
  if (!jsonImportBusy) { fileDragDepth++; jsonImportDialog.classList.add("is-file-drag"); }
});
window.addEventListener("dragover", (event) => {
  if (!jsonImportDialog.open || !draggingFile(event)) return;
  event.preventDefault(); event.dataTransfer.dropEffect = jsonImportBusy ? "none" : "copy";
});
window.addEventListener("dragleave", (event) => {
  if (!jsonImportDialog.open || !fileDragDepth) return;
  fileDragDepth = Math.max(0, fileDragDepth - 1);
  if (!fileDragDepth) clearImportDrag();
});
window.addEventListener("drop", async (event) => {
  if (!jsonImportDialog.open || !draggingFile(event)) return;
  event.preventDefault();
  await loadImportFile([...event.dataTransfer.files]);
});
window.addEventListener("dragend", clearImportDrag);

$("import-owned-documents").addEventListener("click", () => {
  if (!managerEditing) setExperimentEditing(true);
  openJSONImport("owned-documents");
});
async function importOwnedDocumentsData(data) {
  if (!Array.isArray(data)) throw new Error("Documents JSON must be an array.");
  const result = await api("/experiment-definitions/import", {format: "traceability-experiment-v2", experiment: {
    title: "Document import", question: "Document import", documents: [...experimentDocuments, ...data],
  }});
  experimentDocuments = result.definition.documents;
  renderOwnedDocuments(); updateExperimentState();
}

let reviewerPromptBatch = null, reviewerPromptSequence = 0, reviewerPromptPhase = "answer";
async function loadReviewerPrompt() {
  const sequence = ++reviewerPromptSequence;
  const index = Number($("reviewer-prompt-run").value);
  const batch = reviewerPromptBatch;
  $("copy-reviewer-prompt").disabled = true;
  $("copy-reviewer-prompt").textContent = "Copy to clipboard";
  $("reviewer-prompt-status").textContent = "Loading reviewer prompt…";
  $("reviewer-prompt-text").textContent = "";
  try {
    const run = await api(`/runs/${batch.runs[index].id}`);
    if (sequence !== reviewerPromptSequence) return;
    const evaluation = batch.evaluations?.[index];
    const select = $("reviewer-prompt-fact");
    if (reviewerPromptPhase === "fact") {
      const previous = select.value;
      select.replaceChildren();
      const calls = evaluation?.automatic?.semantic_evaluation?.calls || [];
      for (const call of calls.filter(item => item.phase === "fact")) option(select, String(call.fact_index), `Fact ${call.fact_index + 1}`);
      if ([...select.options].some(item => item.value === previous)) select.value = previous;
    }
    const preview = reviewerPromptPreview(run, evaluation, reviewerPromptPhase, Number(select.value));
    $("reviewer-prompt-status").textContent = preview.status;
    $("reviewer-prompt-text").textContent = preview.text;
    $("copy-reviewer-prompt").disabled = !preview.text;
  } catch (error) {
    if (sequence === reviewerPromptSequence) $("reviewer-prompt-status").textContent = `Cannot load reviewer prompt: ${error.message}`;
  }
}
function openReviewerPrompt(batch, phase = "answer") {
  reviewerPromptPhase = phase;
  $("reviewer-prompt-fact-label").hidden = phase !== "fact";
  $("reviewer-prompt-fact").replaceChildren();
  reviewerPromptBatch = batch;
  const select = $("reviewer-prompt-run");
  select.replaceChildren();
  batch.runs.forEach((run, index) => option(select, String(index), `Run ${index + 1}`));
  $("reviewer-prompt-run-label").hidden = batch.runs.length <= 1;
  $("reviewer-prompt-dialog").showModal();
  loadReviewerPrompt();
}
$("reviewer-prompt-run").addEventListener("change", loadReviewerPrompt);
$("reviewer-prompt-fact").addEventListener("change", loadReviewerPrompt);
$("close-reviewer-prompt").addEventListener("click", () => $("reviewer-prompt-dialog").close());
$("reviewer-prompt-dialog").addEventListener("close", () => { ++reviewerPromptSequence; });

$("copy-reviewer-prompt").addEventListener("click", async () => {
  const text = $("reviewer-prompt-text").textContent;
  if (!text) return;
  try {
    await copyToClipboard(text, reviewerPromptPhase === "fact" ? "Fact reviewer prompt" : "Answer reviewer prompt");
  } catch {
    const selection = window.getSelection(), range = document.createRange();
    range.selectNodeContents($("reviewer-prompt-text"));
    selection.removeAllRanges(); selection.addRange(range);
    $("reviewer-prompt-status").textContent = "Clipboard unavailable. The prompt is selected; copy it manually.";
  }
});
