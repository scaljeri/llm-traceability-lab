// Short UI explanations; never included in model input.
const descriptions = {
  'saved-experiment': 'Select the saved experiment to run. Its question, documents and expected answers are loaded together.',
  'managed-experiment': 'Select an experiment to inspect or edit. Save applies your changes; Cancel discards them.',
  'llm-profile': 'The model server that answers the question. This can differ from the review model.',
  'semantic-evaluator': 'The independent model that checks facts, sources and the answer against the expected answers. Disabled means no model review is performed.',
  temperature: 'Allowed range: 0–2 in steps of 0.2, not a percentage. Use decimal values such as 0.4 or 0.8, not 40 or 80. 0 gives the least sampling variation; higher values allow more variation. Controls variation in generated answers. Lower values favor likely tokens; higher values allow more variation. It does not determine factual correctness.',
  'evaluator-temperature': 'Allowed range: 0–2 in steps of 0.2, not a percentage. Use decimal values such as 0.4 or 0.8, not 40 or 80. Controls variation in the review model’s judgments, independently of the answer model. Lower values usually produce more consistent reviews.',
  'seed-enabled': 'Enable an explicit seed for both the answer model and review model. When off, no seed is sent to either server.',
  seed: 'A starting number for token sampling. The same seed can improve reproducibility, but does not guarantee identical answers. Use the on/off switch inside the field to enable it for both models. When off, no seed is sent.',
  'evaluator-threshold': 'Minimum reviewer confidence, from 0 to 100; default 90. Accepted scores below 100 are highlighted yellow. Lower or missing scores and failed reviews count as Fail. A score of 100 is self-reported certainty, not proof of correctness.',
  repetitions: 'Number of attempts with the same question, document order and settings. Every answer and review is recorded separately.',
  'evaluator-max-tokens': 'Override for this execution only. Leave empty to use the selected model setting (4096 if unset). Maximum generated tokens per review response, not per batch. If the response is cut short, the review fails. Default: 4096.',
  'profile-max-tokens': 'Maximum output tokens per call for this model (128–8192), used for both generation and review when selected. This is not the input/context limit. Thinking tokens may count toward this budget, depending on the server. New models default to 2048. Leave empty to use the existing role defaults: 1024 for generation and 4096 for review.',
  'max-tokens': 'Override for this execution only. Leave empty to use the selected model setting (1024 if unset). Maximum generated tokens per answer. A low limit can cut off the JSON response and cause validation to fail.',
  'run-system-prompt': 'Read-only preview of the system prompt sent to the answer model. Together with the user prompt, it forms the model context.',
  'run-context': 'Read-only preview of the user prompt. Experiment placeholders are filled when you select an experiment. Together with the system prompt, it forms the model context.',
  'batch-run-selection': 'Select the recorded attempt whose Markdown and provenance you want to copy or download. The reports retain the complete recorded input, answer and execution data.',
  'experiment-title': 'Name used to identify this experiment in the selector and reports.',
  question: 'The question sent to the answer model. Expected answers belong in the evaluation reference, not here.',
  'prompt-user': 'Supported wildcards: {{user_question}} inserts the experiment question. {{documents}} is required exactly once and inserts the enabled documents in their configured order, with document references and exact text. Put {{output_json}} in the system prompt for the required output structure. Wildcards are replaced when building the model request; inserted text is not expanded again.',
  'prompt-fact': '{{review_input}} inserts the complete fact-review input as JSON: the question, one generated fact, candidate expected facts, cited document text, citation diagnostics. The response specification belongs in the system prompt. Include it exactly once. The answer-model wildcards are not available here.',
  'prompt-answer': '{{review_input}} inserts the complete answer-review input as JSON: question, expected answer, generated answer text and original documents. Include it exactly once. The answer-model wildcards are not available here.',
  'prompt-system': 'Instructions sent as the system message. A review set shares this system prompt between fact and answer reviews. {{output_json}} inserts the required output specification: descriptive answer JSON for generation, the fact-review schema for facts, or the answer-review contract for answers. Other wildcards belong in user prompts. Expected answers are never inserted into the generator system prompt.',
  'answer-prompt': 'Named system and user instructions used by the answer model. Prompts are managed separately from experiment data and copied into each recorded run.',
  'review-facts-prompt': 'The fact prompt and system prompt from this review set check each generated fact separately. This choice is independent of the answer review prompt.',
  'review-answer-prompt': 'The answer prompt and system prompt from this review set check the final answer. This choice is independent of the facts review prompt.',
  'experiment-json': 'Edit the complete experiment as JSON, or paste a full experiment export. Form and JSON views edit the same draft; Save persists it.',
  'answer-json': 'The descriptive structure requested from the answer model. Facts experiments use the app’s fixed structure; these are not expected answer values.',
  'validation-json': 'Expected field values for legacy custom experiments. An empty string means ignore. Facts experiments use expected facts and answer instead.',
  'semantic-reference': 'Expected facts, supporting document references and answer used only by the review model. Optional facts may be absent; unexpected facts are not allowed.',
  'embedding-profile': 'Server used to create embeddings for legacy vector retrieval. Controlled experiments use their own ordered documents without retrieval.',
  'edit-profile': 'Select a saved server profile to edit, or New profile to add a connection.',
  'profile-name': 'Human-readable model name, frozen independently in each generator/reviewer run and used in reports. This label does not verify the underlying model artifact.',
  'profile-kind': 'LLM profiles generate or review answers. Embedding profiles support legacy vector retrieval.',
  'profile-url': 'Base URL of the server’s OpenAI-compatible API, usually ending in /v1. This is a server address, not a model filename.',
  'profile-model': 'Exact model identifier accepted by the server, normally listed by its /v1/models endpoint.',
  'profile-hardware': 'Optional operator-provided description of the hardware. Recorded for comparing and reproducing runs.',
  'profile-precision': 'Optional precision or quantization label, such as FP16 or Q4_K_M. Describes the model’s stored numerical precision.',
  'profile-timeout': 'Maximum seconds to wait for a server request before reporting an error.',
  'profile-dimensions': 'Number of values returned in an embedding vector. Must match the embedding model.',
  'profile-thinking': 'Requests the server chat template’s thinking mode when supported. Server default sends no override.',
  'profile-revision': 'Optional identifier for the model weights or revision. Operator-provided metadata for reproducibility.',
  'profile-server-revision': 'Optional server software version or build identifier, recorded with runs.',
  'profile-manifest': 'Optional deployment JSON with filename, sha256, architecture, quantization, engine, engine_revision, runtime_configuration and provenance. A manifest alone is declared, not verified against the loaded artifact.',
  'profile-capture-metadata': 'Auto reads server metadata for local inference profiles, including older profiles without this setting. Off opts out. Reads engine/build, model path, context, slots, template and sampling defaults; no generation or model-file hashing. Missing values stay unknown.',
  'profile-artifact-path': 'An existing GGUF file accessible on the application host, not a path on a remote machine. The complete file is hashed only by the explicit registration action.',
  'profile-artifact-pid': 'Optional local Linux server process ID. Verification checks the mapped artifact inode, process start/boot identity and API listener. Process arguments alone cannot verify which artifact is loaded.',
  'profile-trust-identity': 'Requires metadata capture and a trusted deployment providing a live traceability_artifact attestation tied to the loaded artifact. Stock llama.cpp does not supply this extension. Never enable this for static or untrusted metadata.',
  'profile-sha': 'Operator-declared SHA-256 of the model file. This field alone does not verify which artifact the server loaded. Live trusted loaded-artifact verification is separate.',
  'profile-query-prefix': 'Text prepended to queries before embedding, when required by the embedding model. Not used for ordered-document experiments.',
  'profile-document-prefix': 'Text prepended to documents before embedding, when required by the embedding model.',
  'json-import-text': 'Paste experiment or document JSON here. You can also drop a JSON file onto the import dialog.',
  'markdown-content': 'Markdown report for this run, ready to copy. Editing or copying this view does not change the recorded run.',
  'batch-markdown-content': 'Analysis Markdown for the batch, including saved settings, answers and review details. Copy this text for further analysis.',
  'context-title': 'Name for a legacy saved instruction block.',
  'context-content': 'Text of a legacy saved instruction block. Saved experiments use their own context instead.',
  'context-active': 'Enable this legacy saved instruction block where instruction-library contexts are used.',
  'top-k': 'Maximum number of chunks selected in legacy vector retrieval. Ignored for controlled experiments with ordered documents.',
};
const documentDescriptions = {
  document_id: 'Document identifier used in citations and expected sources. Together with version it identifies a source inside this experiment.',
  version: 'Version label used in source references. Copied exactly into citations; not an internal database ID.',
  filename: 'Filename describing the document. The actual text supplied to the model is stored in Document text.',
  title: 'Human-readable document title shown with the source.',
  content: 'Document text supplied verbatim when this document is enabled. Disabled documents remain in the experiment but are not sent to the model.',
};

export function fieldHelp(field) {
  const label = field.labels?.[0]?.cloneNode(true);
  label?.querySelectorAll('input, select, textarea, button, output').forEach((node) => node.remove());
  const title = field.getAttribute('aria-label') || label?.textContent.trim() || field.id || 'Field';
  const documentKey = field.dataset.field || ({'document-id':'document_id', 'edit-document-id':'document_id', 'document-version':'version', 'edit-document-version':'version', 'document-filename':'filename', 'edit-document-filename':'filename', 'document-content':'content', 'edit-document-content':'content'})[field.id];
  const text = descriptions[field.id] || documentDescriptions[documentKey]
    || (field.closest('#owned-documents') && field.type === 'checkbox' ? 'Include this document in model input. Unchecking it keeps it in the experiment but excludes it from the supplied sources.' : null)
    || 'Read or edit this value. Changes to a draft take effect when you save it.';
  return {title, text};
}

export function installFieldHelp(root, openHelp) {
  const decorated = new WeakSet();
  let fieldNumber = 0;
  function decorate(node) {
    const fields = [...(node.matches?.('input, select, textarea') ? [node] : []), ...(node.querySelectorAll?.('input, select, textarea') || [])];
    for (const field of fields) {
      if (field.type === 'hidden' || field.hidden || field.hasAttribute('data-no-help') || decorated.has(field)) continue;
      decorated.add(field);
      const button = document.createElement('button');
      button.type = 'button'; button.className = 'info-button secondary field-info'; button.textContent = 'i';
      const help = fieldHelp(field);
      button.setAttribute('aria-label', `About ${help.title}`);
      button.title = `About ${help.title}`;
      button.setAttribute('aria-haspopup', 'dialog'); button.setAttribute('aria-controls', 'parameter-help');
      button.addEventListener('click', (event) => { event.preventDefault(); openHelp(help); });
      let label = field.labels?.[0];
      if (!label) {
        label = document.createElement('label');
        label.textContent = help.title;
        if (field.id) label.htmlFor = field.id;
        field.before(label);
      }
      // A nested info button must never become the label's associated control.
      if (!field.id) {
        let id;
        do { id = `help-field-${++fieldNumber}`; } while (document.getElementById(id));
        field.id = id;
      }
      label.htmlFor = field.id;
      if (label.contains(field) && field.type !== 'checkbox') {
        let container = field;
        while (container.parentElement !== label) container = container.parentElement;
        container.before(button);
      }
      else label.append(button);
    }
  }
  decorate(root);
  const observer = new MutationObserver((records) => {
    for (const record of records) for (const node of record.addedNodes) if (node.nodeType === 1) decorate(node);
  });
  observer.observe(root, {childList: true, subtree: true});
  return observer;
}
