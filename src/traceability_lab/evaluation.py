"""Bescheiden metrics met expliciete tellers, noemers en onbekende waarden."""

import json

from traceability_lab.db import Database, json_text, now
from traceability_lab.documents import digest
from traceability_lab.experiment_evaluation import automatic_evaluation
from traceability_lab.pipeline import read_run, read_trace
from traceability_lab.reproducibility import reproducibility_record


def ratio(numerator: int, denominator: int) -> dict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def persist_automatic_evaluation(db: Database, run_id: str, *, semantic: dict | None = None) -> None:
    """Write once after generation; later reviews never replace these results."""
    started_at = now()
    run = read_run(db, run_id, include_annotations=False)
    bodies = {
        item["id"]: db.one("SELECT content FROM source_versions WHERE id=?", (item["id"],))["content"]
        for item in run["snapshot"]["corpus"]
    }
    if (run["snapshot"].get("evaluation_reference") or {}).get("kind") == "defined-json-v1":
        from traceability_lab.defined_experiments import evaluate_definition

        result = evaluate_definition(run)
    else:
        result = automatic_evaluation(run, bodies)
    if semantic is not None:
        result["semantic_evaluation"] = semantic
    result["reproducibility"] = reproducibility_record(run, result)
    result.update(run_id=run_id, trace_id=run["trace_id"], provenance_uri=f"/api/runs/{run_id}/provenance")
    result.update(evaluation_started_at=started_at, evaluation_finished_at=now())
    with db.connect() as connection:
        connection.execute("INSERT INTO run_evaluations VALUES (?,?,?)", (run_id, now(), json_text(result)))


def evaluate(db: Database, run_id: str) -> dict:
    run = read_run(db, run_id, include_annotations=False)
    trace = read_trace(db, run["trace_id"])
    expected = set(run["snapshot"]["required_spans"])
    observed = {span["name"] for span in trace}
    source_integrity = 0
    for result in run["context"]:
        version = db.one("SELECT * FROM source_versions WHERE id=?", (result["version_id"],))
        if (
            version["content"][result["start"] : result["end"]] == result["text"]
            and digest(result["text"]) == result["sha256"]
            and digest(version["content"]) == version["sha256"]
        ):
            source_integrity += 1
    reference = run["snapshot"].get("evaluation_reference")
    automatic = db.rows("SELECT payload FROM run_evaluations WHERE run_id=?", (run_id,))
    required = reference["required_evidence"] if reference else []
    found = sum(
        any(
            result["document_id"] == item["document_id"]
            and result["version"] == item["version"]
            and item["quote"] in result["text"]
            for result in run["retrievals"]
        )
        for item in required
    )
    stored_automatic = json.loads(automatic[0]["payload"]) if automatic else None
    result_criteria = None
    if (reference or {}).get("definition", {}).get("output_mode") in (
        "facts_and_answer",
        "facts_and_conclusion",
    ):
        from traceability_lab.fact_answers import fact_result_criteria

        result_criteria = fact_result_criteria(run, stored_automatic)
    # Citation precision en claim-support recall vragen een aparte, volledig gemapte referentieset.
    # Ze als 0 rapporteren zou onbekend verwarren met een gemeten mislukking.
    from traceability_lab.performance import run_performance

    return {
        "reproducibility": (stored_automatic or {}).get("reproducibility"),
        "performance": run_performance(run, trace, stored_automatic),
        "schema": "evaluation-v1",
        "backend": run["backend"],
        "research_output": run["backend"] == "local" and run["status"] == "completed",
        "reference_sha256": digest(json_text(reference)) if reference else None,
        "reference_status": "frozen" if "evaluation_reference" in run["snapshot"] else "not_recorded",
        "reference": reference,
        "automatic": stored_automatic,
        "result_criteria": result_criteria,
        "result_criteria_version": "facts-results-v8" if result_criteria is not None else None,
        "retrieval_recall_at_k": ratio(found, len(required))
        if run["snapshot"].get("context_mode", "retrieval") == "retrieval"
        and not run["snapshot"].get("context_prepared")
        and run["question"] == (run["snapshot"].get("case") or {}).get("question", run["question"])
        else None,
        "context_source_integrity": ratio(source_integrity, len(run["context"])),
        "trace_stage_coverage": ratio(len(expected & observed), len(expected)),
        "missing_trace_stages": sorted(expected - observed),
        "retrieved_source_integrity": ratio(source_integrity, len(run["context"]))
        if run["snapshot"].get("context_mode", "retrieval") == "retrieval"
        else None,
        "citation_precision": None,
        "claim_support_recall": None,
        "unavailable_reason": "Citation precision and claim-support recall are not calculated by this build. See deterministic citation checks and semantic reviewer judgments for recorded results.",
    }


def batch_summary(db: Database, batch_id: str, *, evaluations_cache: dict | None = None) -> dict:
    runs = db.rows(
        "SELECT id,status,backend,answer,error,json_extract(snapshot,'$.json_repair.status') AS json_repair_status, "
        "json_extract(snapshot,'$.experiment.planned_repetitions') "
        "AS planned_repetitions,COALESCE(json_extract(snapshot,'$.experiment.definition.title'),"
        "json_extract(snapshot,'$.case.title')) AS experiment_title FROM runs WHERE batch_id=? "
        "AND NOT EXISTS (SELECT 1 FROM run_evaluations e WHERE e.run_id=runs.id "
        "AND json_extract(e.payload,'$.semantic_evaluation.status')='cancelled') ORDER BY created_at",
        (batch_id,),
    )
    if not runs:
        raise LookupError("Batch not found.")
    evaluations = []
    for run in runs:
        identity = run["id"]
        if evaluations_cache is not None and identity in evaluations_cache:
            result = evaluations_cache[identity]
        else:
            result = evaluate(db, identity)
            if evaluations_cache is not None:
                evaluations_cache[identity] = result
        evaluations.append(result)
    from traceability_lab.performance import experiment_overview

    lifecycle = db.one("SELECT * FROM batch_executions WHERE id=?", (batch_id,))
    return {
        "overview": experiment_overview([item["performance"] for item in evaluations], lifecycle),
        "batch_id": batch_id,
        "experiment_title": runs[0]["experiment_title"],
        "runs": runs,
        "planned_total": max([len(runs), *(run["planned_repetitions"] or 0 for run in runs)]),
        "counts": {
            "total": len(runs),
            "completed": sum(run["status"] == "completed" for run in runs),
            "failed": sum(run["status"] == "failed" for run in runs),
        },
        "property_counts": property_counts(evaluations),
        "evaluations": evaluations,
    }


def property_counts(evaluations: list[dict]) -> list[dict]:
    """Tel elke poging mee; onbekend of onbeoordeeld wordt geen succes."""
    rows = [
        item.get("result_criteria")
        if item.get("result_criteria") is not None
        else (item.get("automatic") or {}).get("criteria", [])
        for item in evaluations
    ]
    keys = dict.fromkeys(
        criterion["id"] for row in rows for criterion in row if criterion.get("method") != "ignored"
    )
    result = []
    for key in keys:
        counts = {"pass": 0, "fail": 0, "requires_review": 0, "not_evaluated": 0}
        for row in rows:
            criterion = next((item for item in row if item["id"] == key), None)
            state = (
                criterion["result"]
                if criterion and criterion["applicability"] == "applicable"
                else "not_evaluated"
            )
            if state == "evaluation_error":
                counts.setdefault("evaluation_error", 0)
            counts[state if state in counts else "not_evaluated"] += 1
        result.append({"id": key, "total": len(evaluations), **counts})
    return result
