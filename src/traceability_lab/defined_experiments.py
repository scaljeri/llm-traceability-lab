"""Eigen gecontroleerde experimenten en expliciete JSON-veldcontroles.

Answer JSON is een vormvoorbeeld, geen JSON Schema en geen ground truth.
Validation JSON bevat verwachte waarden; $source verwijst naar aangeboden bewijs.
"""

import json
import math
from uuid import uuid4

from traceability_lab.db import Database, json_text, now
from traceability_lab.documents import digest
from traceability_lab.experiment_evaluation import parse_answer


def validate_contract(answer: dict, validation: dict) -> None:
    """Weiger onbruikbare of niet-eindige verwachtingen vóór een run."""
    if len(json_text([answer, validation])) > 40_000:
        raise ValueError("Answer JSON and Validation JSON exceed 40,000 characters.")

    def check(template, expected):
        if isinstance(expected, dict) and "$source" in expected:
            source = expected["$source"]
            if set(expected) != {"$source"} or not isinstance(source, dict):
                raise ValueError("Use {$source: {document_id: ..., version: ...}}.")
            if set(source) != {"document_id", "version"} or not all(
                isinstance(value, str) and value.strip() for value in source.values()
            ):
                raise ValueError("$source requires string document_id and version.")
        elif isinstance(expected, dict):
            if not isinstance(template, dict) or not expected.keys() <= template.keys():
                raise ValueError("Validation fields must exist in Answer JSON.")
            for key, value in expected.items():
                check(template[key], value)

    check(answer, validation)


def list_definitions(db: Database) -> list[dict]:
    from traceability_lab.models import SemanticReference
    from traceability_lab.owned_documents import copy_legacy_documents

    rows = db.rows(
        "SELECT * FROM experiment_definitions ORDER BY json_extract(definition, '$.title') COLLATE NOCASE, id"
    )
    return [
        {
            **row,
            "document_copies": copy_legacy_documents(db, json.loads(row["definition"])["version_ids"])
            if json.loads(row["definition"]).get("documents") is None
            else None,
            "definition": {
                "system_prompt": "",
                "output_mode": "custom",
                "semantic_reference": None,
                "semantic_evaluator": None,
                **json.loads(row["definition"]),
                **(
                    {"output_mode": "facts_and_answer"}
                    if json.loads(row["definition"]).get("output_mode") == "facts_and_conclusion"
                    else {}
                ),
                **(
                    {
                        "semantic_reference": SemanticReference.model_validate(
                            json.loads(row["definition"])["semantic_reference"]
                        ).model_dump()
                    }
                    if json.loads(row["definition"]).get("semantic_reference") is not None
                    else {}
                ),
            },
        }
        for row in rows
    ]


def save_definition(db: Database, definition, identity: str | None = None) -> dict:
    from traceability_lab.pipeline import selected_versions

    if definition.documents is None:
        selected_versions(db, definition.version_ids)
    with db.connect() as connection:
        if identity:
            cursor = connection.execute(
                "UPDATE experiment_definitions SET definition=?,revision=revision+1,updated_at=? WHERE id=?",
                (json_text(definition.model_dump()), now(), identity),
            )
            if not cursor.rowcount:
                raise LookupError("Experiment not found.")
        else:
            identity = "saved-" + uuid4().hex
            timestamp = now()
            connection.execute(
                "INSERT INTO experiment_definitions VALUES (?,?,?,?,?)",
                (identity, 1, json_text(definition.model_dump()), timestamp, timestamp),
            )
    return next(item for item in list_definitions(db) if item["id"] == identity)


def delete_definition(db: Database, identity: str) -> dict:
    """Remove the editable definition; immutable run snapshots remain available."""
    with db.connect() as connection:
        cursor = connection.execute("DELETE FROM experiment_definitions WHERE id=?", (identity,))
        if not cursor.rowcount:
            raise LookupError("Experiment not found.")
    return {"id": identity, "deleted": True}


def prepare_definition_runs(db, settings, embedder, tracing, request, *, cancel_event=None, batch_id=None):
    """Valideer vóór streaming; stuur voortgang pas na opgeslagen evaluatie."""
    from traceability_lab.evaluation import persist_automatic_evaluation
    from traceability_lab.models import QuestionInput
    from traceability_lab.pipeline import run_question, selected_versions
    from traceability_lab.profiles import resolve_profiles

    definition = request.definition.model_dump()
    execution = request.execution
    from traceability_lab.prompts import read_prompt

    selected_prompts = {}
    if execution.answer_prompt_id:
        prompt = read_prompt(db, execution.answer_prompt_id, "answer")
        selected_prompts["answer"] = prompt
        definition["system_prompt"] = prompt["system_prompt"]
        definition["context"] = prompt["user_prompt"]
    for stage in ("facts", "answer"):
        identity = getattr(execution, f"review_{stage}_prompt_id")
        if identity:
            selected_prompts[f"review_{stage}"] = read_prompt(db, identity, "review")
    for prompt in selected_prompts.values():
        prompt["sha256"] = digest(json_text(prompt))
    if request.definition.documents is not None:
        from traceability_lab.owned_documents import owned_sources

        selected, _ = owned_sources(db, request.definition.documents)
    else:
        selected = selected_versions(db, definition["version_ids"])
    question = QuestionInput(
        **{
            key: getattr(execution, key)
            for key in (
                "mode",
                "temperature",
                "seed",
                "max_tokens",
                "llm_profile_id",
                "embedding_profile_id",
                "llm_context",
                "context_chunk_ids",
            )
            # Named templates are rendered on the server from the frozen selection.
            # A stale browser preview must never override the selected prompt.
            if not ("answer" in selected_prompts and key in {"llm_context", "context_chunk_ids"})
        },
        system_prompt=selected_prompts["answer"]["system_prompt"]
        if "answer" in selected_prompts
        else execution.system_prompt
        if execution.system_prompt is not None
        else definition["system_prompt"] or None,
        documents=request.definition.documents,
        output_mode=definition["output_mode"],
        question=definition["question"],
        version_ids=definition["version_ids"],
        context_mode="explicit",
        context_ids=[],
        prompt_instructions=definition["context"],
        answer_json=definition["answer_json"],
    )
    settings, profiles = resolve_profiles(db, settings, question)
    saved = next((item for item in list_definitions(db) if item["id"] == request.saved_id), None)
    if request.saved_id and saved is None:
        raise LookupError("Experiment not found.")
    reference = {
        "kind": "defined-json-v1",
        "definition": definition,
        "definition_sha256": digest(json_text(definition)),
        "required_evidence": [],
        "expected": json_text(definition["validation_json"]),
        "source_hashes": {item["id"]: item["sha256"] for item in selected},
    }
    experiment = {
        "id": request.saved_id,
        "definition_version": saved["revision"] if saved else "draft",
        "definition": definition,
        "modified": not saved or saved["definition"] != request.definition.model_dump(),
        "planned_repetitions": execution.repetitions,
        "ground_truth": reference,
        "prompts": selected_prompts,
    }
    from traceability_lab.semantic_evaluation import prepare_evaluator

    evaluator = prepare_evaluator(
        db,
        settings,
        definition,
        fact_prompt=selected_prompts.get("review_facts"),
        answer_prompt=selected_prompts.get("review_answer"),
    )
    if evaluator:
        experiment["semantic_evaluator_snapshot"] = evaluator["metadata"]
    batch_id = batch_id or uuid4().hex

    def events():
        from traceability_lab.evaluation import batch_summary
        from traceability_lab.retention import retained_batch

        with retained_batch(db, batch_id):
            ids = []
            evaluations_cache = {}
            yield {"type": "started", "batch_id": batch_id, "total": execution.repetitions}
            try:
                for index in range(execution.repetitions):
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    identity = run_question(
                        db,
                        settings,
                        embedder,
                        tracing,
                        question,
                        batch_id=batch_id,
                        profiles_snapshot=profiles,
                        contexts_snapshot=[],
                        experiment_snapshot=experiment,
                    )
                    from traceability_lab.semantic_evaluation import observe_semantic_evaluation

                    if cancel_event is not None and cancel_event.is_set():
                        persist_automatic_evaluation(
                            db,
                            identity,
                            semantic={"status": "cancelled", "reason": "Cancelled by user; review skipped."},
                        )
                        break
                    semantic = (
                        observe_semantic_evaluation(db, tracing, identity, evaluator) if evaluator else None
                    )
                    cancelled = cancel_event is not None and cancel_event.is_set()
                    if cancelled:
                        semantic = {
                            **(semantic or {}),
                            "status": "cancelled",
                            "reason": "Cancelled by user; incomplete run excluded from results.",
                        }
                    if semantic is None:
                        persist_automatic_evaluation(db, identity)
                    else:
                        persist_automatic_evaluation(db, identity, semantic=semantic)
                    if cancelled:
                        break
                    ids.append(identity)
                    yield {
                        "type": "progress",
                        "batch_id": batch_id,
                        "completed": index + 1,
                        "total": execution.repetitions,
                        "run_id": identity,
                        "summary": batch_summary(db, batch_id, evaluations_cache=evaluations_cache),
                    }
            except Exception as error:
                from traceability_lab.cancellation import RunCancelled

                if isinstance(error, RunCancelled) and cancel_event is not None and cancel_event.is_set():
                    yield {
                        "type": "cancelled",
                        "batch_id": batch_id,
                        "run_ids": ids,
                        "summary": batch_summary(db, batch_id, evaluations_cache=evaluations_cache)
                        if ids
                        else None,
                    }
                    return
                yield {
                    "type": "error",
                    "batch_id": batch_id,
                    "run_ids": ids,
                    "message": f"{type(error).__name__}: {error}",
                    "summary": batch_summary(db, batch_id, evaluations_cache=evaluations_cache)
                    if ids
                    else None,
                }
                return
        yield {
            "type": "cancelled" if cancel_event is not None and cancel_event.is_set() else "completed",
            "batch_id": batch_id,
            "run_ids": ids,
            "summary": batch_summary(db, batch_id, evaluations_cache=evaluations_cache) if ids else None,
        }

    return events()


def run_definition(db, settings, embedder, tracing, request) -> dict:
    """Behoud de bestaande synchrone API naast de streamingroute."""
    result = None
    for result in prepare_definition_runs(db, settings, embedder, tracing, request):
        if result["type"] == "error":
            raise ValueError(result["message"])
    return {"batch_id": result["batch_id"], "run_ids": result["run_ids"]}


def evaluate_definition(run: dict) -> dict:
    """Exacte veldwaarden; broncontrole bewijst geen causale bronbenutting."""
    if run["snapshot"]["evaluation_reference"]["definition"].get("output_mode") == "facts_and_answer":
        from traceability_lab.fact_answers import evaluate_fact_answer

        return evaluate_fact_answer(run)
    reference = run["snapshot"]["evaluation_reference"]
    definition = reference["definition"]
    parsed = parse_answer(run.get("answer"))
    truncated = (run.get("response") or {}).get("choices", [{}])[0].get("finish_reason") == "length"
    usable = run["backend"] != "demo" and run["status"] == "completed"
    criteria = []

    def add(key, result, reason, *, applicability="applicable", method="deterministic", **details):
        criteria.append(
            {
                "id": key,
                "applicability": applicability,
                "result": result,
                "method": method,
                "reason": reason,
                **details,
            }
        )

    from traceability_lab.json_repair import matches_structure

    valid = parsed is not None and not truncated
    format_ok = valid and matches_structure(definition["answer_json"], parsed)
    add(
        "answer_json",
        ("pass" if format_ok else "fail") if usable and not truncated else "requires_review",
        "Checks JSON object structure and required keys; template scalar values are placeholders.",
    )

    def equal(actual, expected):
        if expected == "":
            return True
        if isinstance(actual, bool) or isinstance(expected, bool):
            return type(actual) is type(expected) and actual == expected
        if isinstance(expected, (int, float)):
            return (
                type(actual) in (int, float)
                and (type(actual) is int or math.isfinite(actual))
                and actual == expected
            )
        if isinstance(expected, list):
            return (
                isinstance(actual, list)
                and len(actual) == len(expected)
                and all(equal(a, b) for a, b in zip(actual, expected, strict=True))
            )
        if isinstance(expected, dict):
            return (
                isinstance(actual, dict)
                and actual.keys() == expected.keys()
                and all(equal(actual[key], value) for key, value in expected.items())
            )
        return type(actual) is type(expected) and actual == expected

    def all_ignored(value):
        if value == "":
            return True
        children = value.values() if isinstance(value, dict) else value if isinstance(value, list) else []
        return bool(value) and isinstance(value, (dict, list)) and all(all_ignored(item) for item in children)

    def check(expected, actual, path):
        for key, value in expected.items():
            label = f"{path}.{key}" if path else key
            present = isinstance(actual, dict) and key in actual
            observed = actual.get(key) if isinstance(actual, dict) else None
            if value == "" or (isinstance(value, list) and all_ignored(value)):
                add(
                    label,
                    "not_evaluated",
                    "Ignored: empty expected strings disable value checks.",
                    applicability="not_applicable",
                    method="ignored",
                    expected=value,
                    observed=observed,
                    present=present,
                )
                continue
            if isinstance(value, dict) and "$source" not in value:
                check(value, observed, label)
                continue
            source_check = isinstance(value, dict) and "$source" in value
            if source_check:
                source = next(
                    (
                        item
                        for item in run["context"]
                        if isinstance(observed, str)
                        and item["chunk_id"] == observed.removeprefix("[").removesuffix("]")
                    ),
                    None,
                )
                matches = source is not None and all(
                    source[field] == wanted for field, wanted in value["$source"].items()
                )
            else:
                matches = present and equal(observed, value)
            result = "pass" if matches else "fail"
            reason = (
                "Matches a supplied document/version citation; does not verify claim entailment."
                if source_check
                else "Exact expected JSON value (numeric notation may vary)."
            )
            if not usable or not valid:
                result = "requires_review"
            if source_check and run["snapshot"].get("context_edited"):
                result = "requires_review"
                reason = "Edited context has no verified mapping to the original source chunks."
            add(label, result, reason, expected=value, observed=observed, present=present)

    check(definition["validation_json"], parsed, "")
    return {
        "evaluator_version": "defined-json-v2",
        "criteria": criteria,
        "reference_sha256": digest(json_text(reference)),
    }
