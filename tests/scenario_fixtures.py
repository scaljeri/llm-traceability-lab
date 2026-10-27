"""Test-only historical scenarios; never loaded by the application."""

import json
from pathlib import Path
from uuid import uuid4

from traceability_lab.config import Settings
from traceability_lab.contexts import resolve_contexts
from traceability_lab.db import Database
from traceability_lab.documents import ingest, version_chunks
from traceability_lab.embeddings import Embedder
from traceability_lab.evaluation import persist_automatic_evaluation
from traceability_lab.models import DocumentInput, ExperimentInput, QuestionInput
from traceability_lab.pipeline import context_text, default_system_prompt, run_question, selected_versions
from traceability_lab.profiles import resolve_profiles
from traceability_lab.tracing import Tracing

FIXTURES = Path(__file__).parent / "fixtures"
REFERENCE = FIXTURES / "scenario_reference.json"


def reference_case(case_id):
    return json.loads(REFERENCE.read_text())["cases"].get(case_id)


def bootstrap(db: Database):
    manifest = json.loads((FIXTURES / "documents/manifest.json").read_text())
    for entry in manifest["documents"]:
        if db.rows(
            "SELECT id FROM source_versions WHERE document_id=? AND version=?",
            (entry["document_id"], entry["version"]),
        ):
            continue
        content = (FIXTURES / "documents" / entry["file"]).read_bytes().decode("utf-8")
        ingest(
            db,
            DocumentInput(
                document_id=entry["document_id"],
                title=entry["title"],
                version=entry["version"],
                source=f"orion-fixture-v1/{entry['file']}",
                content=content,
                metadata={"dataset": "orion-fixture-v1", "fictional": "true"},
            ),
        )


def cases() -> list[dict]:
    return json.loads((FIXTURES / "scenarios.json").read_text())


def case_by_id(case_id: str) -> dict:
    for case in cases():
        if case["id"] == case_id:
            return case
    raise LookupError("Unknown experiment.")


def run_experiment(
    db: Database,
    settings: Settings,
    embedder: Embedder,
    tracing: Tracing,
    case_id: str,
    request: ExperimentInput,
) -> dict:
    case = case_by_id(case_id)
    versions = request.version_ids
    if versions is None:
        versions = [
            db.one(
                "SELECT id FROM source_versions WHERE document_id=? AND version=? AND deleted_at IS NULL",
                tuple(pair),
            )["id"]
            for pair in case["versions"]
        ]
    defaults = {
        key: case[key]
        for key in (
            "question",
            "context_mode",
            "top_k",
            "prompt_instructions",
            "version_instructions",
            "mode",
            "temperature",
            "seed",
            "max_tokens",
        )
    }
    effective = {
        **defaults,
        **{
            key: value
            for key, value in request.model_dump(exclude_unset=True).items()
            if key in defaults and value is not None
        },
    }
    question = QuestionInput(
        system_prompt=request.system_prompt,
        llm_context=request.llm_context,
        context_chunk_ids=request.context_chunk_ids,
        context_override=request.context_override,
        context_ids=request.context_ids,
        version_ids=versions,
        llm_profile_id=request.llm_profile_id,
        embedding_profile_id=request.embedding_profile_id,
        **effective,
    )
    selected = selected_versions(db, question.version_ids)
    settings, profiles = resolve_profiles(db, settings, question)
    supplemental = resolve_contexts(db, request.context_ids, request.context_override)
    actual = {
        "context_override": request.context_override,
        **effective,
        "versions": [[row["document_id"], row["version"]] for row in selected],
        "context_ids": [item["id"] for item in supplemental],
    }
    original = {**defaults, "versions": case["versions"], "context_ids": [], "context_override": None}
    if request.system_prompt is not None and request.system_prompt != default_system_prompt(question):
        original["system_prompt"] = default_system_prompt(question)
        actual["system_prompt"] = request.system_prompt
    if request.llm_context is not None:
        chunks = version_chunks(db, versions)
        by_id = {chunk["id"]: chunk for chunk in chunks}
        preview_chunks = [by_id[key] for key in (request.context_chunk_ids or []) if key in by_id]
        if request.llm_context != context_text(question, preview_chunks, supplemental):
            actual["llm_context"] = request.llm_context
            original["llm_context"] = None
    overrides = {key: value for key, value in actual.items() if value != original.get(key)}
    experiment = {
        "id": case_id,
        "definition_version": case["definition_version"],
        "definition": case,
        "original_defaults": original,
        "actual_configuration": actual,
        "overrides": overrides,
        "modified": bool(overrides),
        "ground_truth": reference_case(case_id),
    }
    batch_id = uuid4().hex
    from traceability_lab.retention import retained_batch

    with retained_batch(db, batch_id):
        run_ids = []
        for _ in range(request.repetitions):
            run_id = run_question(
                db,
                settings,
                embedder,
                tracing,
                question,
                case=case,
                batch_id=batch_id,
                profiles_snapshot=profiles,
                contexts_snapshot=supplemental,
                experiment_snapshot=experiment,
            )
            persist_automatic_evaluation(db, run_id)
            run_ids.append(run_id)
    return {"batch_id": batch_id, "run_ids": run_ids}


def install_scenarios(app, settings):
    """Attach test endpoints after creating an otherwise empty production app."""
    from traceability_lab.db import Database

    db = Database(settings.database)

    @app.get("/test-scenarios")
    def scenarios():
        return cases()

    @app.post("/test-scenarios/{case_id}/run", status_code=201)
    def execute(case_id: str, data: ExperimentInput):
        if "repetitions" not in data.model_fields_set:
            data = data.model_copy(update={"repetitions": 1})
        return run_experiment(db, settings, app.state.embedder, app.state.tracing, case_id, data)
