// Pure editor helpers. Validation values never enter generated instructions.
export function blankValidation(value) {
  if (Array.isArray(value)) return value.map(blankValidation);
  if (value && typeof value === "object") return Object.fromEntries(
    Object.entries(value).map(([key, item]) => [key, blankValidation(item)]));
  return "";
}

export function validationIsEmpty(text) {
  if (!text.trim()) return true;
  try {
    const value = JSON.parse(text);
    return value && typeof value === "object" && !Array.isArray(value) && !Object.keys(value).length;
  } catch { return false; } // Do not overwrite an unfinished or invalid draft.
}

export const factAnswerTemplate = {"facts": [{"fact": "A relevant fact from the supplied documents, expressed as a natural-language statement. Include only facts relevant to answering the research question.", "sources": [{"document_id": "The exact document_id of the document supporting this fact.", "version": "The exact version of the document supporting this fact.", "quote": "An exact, contiguous quotation from the source document that supports this fact. Copy the original text verbatim without paraphrasing."}]}], "answer": "A natural-language answer to the research question, based only on the extracted facts. Explain uncertainty or conflicting evidence when relevant, following the task instructions. Use null when no supported answer can be stated; uncertainty alone does not require null."};

export function definitionDefaults(value) {
  value = structuredClone(value || {});
  if (value.output_mode === "facts_and_conclusion") value.output_mode = "facts_and_answer";
  if (value.semantic_reference && !("answer" in value.semantic_reference) && "conclusion" in value.semantic_reference) {
    value.semantic_reference.answer = value.semantic_reference.conclusion;
    delete value.semantic_reference.conclusion;
  }
  return {documents: null, system_prompt: "", output_mode: "custom", semantic_reference: null, semantic_evaluator: null, ...value};
}

export function experimentJSONValue(definition) {
  const value = structuredClone(definition);
  delete value.version_ids;
  delete value.system_prompt;
  delete value.context;
  if (value.output_mode === 'facts_and_answer') {
    delete value.output_mode;
    delete value.answer_json;
    delete value.validation_json;
  }
  if (definition.output_mode === 'facts_and_answer') {
    delete value.semantic_evaluator;
  }
  return value;
}

export function parseExperimentJSON(text) {
  const value = JSON.parse(text);
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Use an experiment JSON object.');
  const experiment = value.format === 'traceability-experiment-v2' ? value.experiment : value;
  if (!experiment || typeof experiment !== 'object' || Array.isArray(experiment)) throw new Error('Use an experiment JSON object.');
  return definitionDefaults({output_mode: 'facts_and_answer', system_prompt: '', ...experiment});
}

export function experimentChanged(saved, draft) {
  const canonical = (value) => Array.isArray(value) ? value.map(canonical)
    : value && typeof value === "object" ? Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])])) : value;
  const comparable = (value) => canonical(experimentJSONValue(definitionDefaults(value)));
  return JSON.stringify(comparable(saved)) !== JSON.stringify(comparable(draft));
}

// One-time transition from the old default; subsequent explicit choices remain intact.
export function reviewSettings(config, migrated) {
  const threshold = config?.confidence_threshold ?? 90;
  return {...config, confidence_threshold: !migrated && threshold === 100 ? 90 : threshold};
}

export function outputTokenLimit(override, profile, fallback) {
  return override === "" ? (profile?.max_tokens ?? fallback) : Number(override);
}
