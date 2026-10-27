# Experiment JSON format

Each experiment owns its documents, including disabled ones. Import/export is a
single JSON object containing the configuration and document text. Array order
sets the order of enabled documents in the generator input. `?` means optional.
These types describe portable JSON, not internal database records.

## Types

```typescript
type JsonValue =
  | string | number | boolean | null
  | JsonValue[]
  | { [key: string]: JsonValue };
type JsonObject = { [key: string]: JsonValue };

interface ExperimentFile {
  format: "traceability-experiment-v2";
  experiment: Experiment;
  source_ids?: Record<string, string[]>;
}

interface Experiment {
  title: string;
  question: string;
  documents: ExperimentDocument[];
  output_mode?: "custom" | "facts_and_answer";
  // The app supplies the fixed descriptive output structure.
  validation_json?: JsonObject;
  semantic_reference?: SemanticReference | null;
  semantic_evaluator?: SemanticEvaluator | null;
}

interface ExperimentDocument {
  document_id: string;
  version: string;
  content: string;
  enabled?: boolean;
  title?: string;
  filename?: string | null;
  source?: string;
  metadata?: Record<string, string>;

  // Optional transfer metadata, not editable document fields.
  id?: string | null;
  sha256?: string | null;
}

interface SemanticReference {
  facts?: Array<string | { fact: string; sources: Array<{ document_id: string; version: string; quote?: string }>; optional?: boolean }>;
  answer?: string | null;
}

interface SemanticEvaluator {
  confidence_threshold?: number | null; // 0–100; default 90. null: legacy unscored judgments.
  llm_profile_id: string;
  temperature?: number;
  seed?: number | null;
  max_tokens?: number;
}
```

## Fields

| Field | Meaning |
| --- | --- |
| `format` | Use `traceability-experiment-v2` for self-contained experiments. |
| `title` | Experiment name, up to 200 characters. |
| `question` | Question for the generator, up to 2,000 characters. |
| `documents` | Up to 100 owned documents, in input order. Only enabled documents are supplied. An empty array is allowed. |
| `output_mode` | Use `facts_and_answer` for new experiments. `custom` is retained for compatibility. |
| `validation_json` | Use `{}` for facts experiments. Legacy custom validation compares expected field values; `""` means ignore. |
| `semantic_reference` | Expected facts and answer, used only by the reviewer; never inserted into the generator prompt. |
| `semantic_evaluator` | Optional saved reviewer settings. Profiles are local and must exist on the importing installation. The runner lets you select your own reviewer. |
| `source_ids` | Optional exported document-reference to internal fragment-ID mapping. Changes on import produce warnings, not a shared document library. |

## Documents

| Field | Meaning |
| --- | --- |
| `document_id` | User-chosen identifier, up to 100 characters: letters, numbers, `_` and `-`. ID + version must be unique within the experiment. |
| `version` | Version label, up to 80 characters; default `"1"`. Export uses a string. |
| `content` | Exact document text, up to 100,000 characters per document. Enabled documents require nonempty text. |
| `enabled` | Defaults to `true`. Disabled documents stay in the experiment but are excluded from the input. |
| `title` | Optional title, up to 200 characters; derived when empty. |
| `filename` | Optional filename, up to 500 characters; derived when empty. |
| `source` | Optional source label, up to 500 characters. |
| `metadata` | Optional string-to-string metadata; defaults to `{}`. |
| `id` | Optional transfer reference such as `orion@1`, matching document ID and URL-encoded version. Not a database ID. |
| `sha256` | Optional SHA-256 of exact UTF-8 content, 64 lowercase hexadecimal characters; verified on import. |

Enabled document text together must fit within 100,000 characters. Two experiments
may reuse an ID and version with different text. Editing one does not change the
other or the source records captured in historical runs. A version label does not
automatically establish authority or approval.

## Reference and reviewer settings

`semantic_reference.facts` accepts up to 100 expected statements, each up to
5,000 characters. New entries use `{fact, sources, optional}`. `sources` identifies
the expected document/version pairs and may include exact reference quotations.
Old string-only facts remain readable. `answer` is an expected answer of up to
10,000 characters, or `null` when no answer can be determined.

A missing reference means no expected-answer comparison. A reference object with
`answer: null` explicitly expects absence of a determinable answer.

Reviewer settings use a local `llm_profile_id`, independent `temperature` (0–2,
default 0), optional `seed` (0–2,147,483,647, default off), `max_tokens` (128–8,192,
default 4,096), and `confidence_threshold` (0–100, default 90). A null threshold
is retained for historical unscored reviews. The model profile is not bundled.

## Example

```json
{
  "format": "traceability-experiment-v2",
  "experiment": {
    "title": "Example experiment",
    "question": "What does the evidence establish?",
    "validation_json": {},
    "documents": [
      {
        "document_id": "example",
        "version": "1",
        "content": "Document text.",
        "enabled": true
      },
      {
        "document_id": "alternative",
        "version": "1",
        "content": "Alternative text excluded from this run.",
        "enabled": false
      }
    ]
  }
}
```

## Editing and compatibility

Import a complete experiment file by pasting JSON or dropping a file into the
import dialog. Form and JSON views edit the same draft. Save persists the draft;
export includes its ordered documents, including disabled entries. The examples
under `experiments/` show complete facts experiments.

For new experiments, use `output_mode: "facts_and_answer"`, `validation_json: {}`
and put expectations in `semantic_reference`. The app supplies the fixed output
structure; do not add `answer_json`. Its output is:

```typescript
interface GeneratedAnswer {
  facts: Array<{
    fact: string;
    sources: Array<{ document_id: string; version: string; quote: string }>;
  }>;
  answer: string | null;
}
```

Expected facts are exhaustive. Each fact is required unless `optional: true`.
Optional facts can be absent; returned facts still need correct meaning, expected
sources and valid literal quotations. Fact order is not significant. Expected
source references also support `quote`; keep it exact when provided.

`semantic_evaluator.llm_profile_id` refers to a profile on the importing installation.
Profiles are not bundled with experiments. Choose the generator and reviewer in
the runner; execution settings are frozen for the batch. The confidence threshold
is 0–100 and defaults to 90. Reviewer temperature is independent of generator
sampling. See the result notes below for judgment and acceptance rules.

Legacy `custom` output and `validation_json` remain supported for compatibility.
The importer also accepts the old `conclusion` field as an alias for `answer`.
Experiments contain data and expected results; prompts are managed separately.
Answer prompts support only `{{user_question}}`, `{{documents}}` and `{{output_json}}`.
Historical records are not rewritten when importing or editing an experiment.

Prompts belong in the prompt library, not experiment JSON. The API's legacy
`context` and `system_prompt` fields remain a fallback for custom executions
without a selected generation prompt; a selected prompt takes precedence.

## Reading results

The app checks output format, facts with sources, and the final answer separately.
A final format failure skips semantic review and fails the dependent checks.
Each returned fact is reviewed individually against the expected facts and its
cited documents. The answer is reviewed separately with the question, expected
answer and original documents. Expected values never enter the generator prompt.

Meaning, semantic correctness, source support and reviewer confidence are distinct
judgments. Correctness and confidence both use 0–100. Acceptance depends on the
recorded judgment and confidence threshold as well as deterministic checks.
A source must match the expected document/version references, and a quotation must
be an exact contiguous substring of the supplied text. Formatting or whitespace
differences are not silently repaired or accepted as valid citations.

Returned facts outside the exhaustive reference fail even when supported by a
document. Required facts must be covered; optional facts can be omitted. The
order of returned facts is free. An expected `answer: null` represents the absence
of a determinable answer, not permission to replace a specific expected answer
with null.

Expand run results for individual judgments and rejection reasons. Reviewer prompt
dialogs show recorded inputs and support copying. Analysis and run exports retain
original responses, prompts, source text and settings. Later edits do not regrade
historical runs. Model judgments can be wrong; inspect results when comparing
experiments rather than treating the reviewer as ground truth.

## Prompt sets

Answer and review prompts are separate from experiment data. See the
[prompt JSON formats](prompts.md) for import/export and placeholders.
