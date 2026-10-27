"""Voortgang telt opgeslagen pogingen; grafiektotalen houden review apart."""

import json

from traceability_lab.defined_experiments import prepare_definition_runs
from traceability_lab.evaluation import property_counts
from traceability_lab.models import DefinedExperimentRun, ExperimentInput


def request_data(client, repetitions=3):
    sources = [doc["id"] for doc in client.get("/api/documents").json() if doc["document_id"] == "orion"]
    return {
        "definition": {
            "title": "Progress experiment",
            "question": "What is the budget?",
            "version_ids": sources,
            "context": "Return JSON with budget_eur.",
            "answer_json": {"budget_eur": "number or null"},
            "validation_json": {"budget_eur": 2800000},
        },
        "execution": {"llm_profile_id": "test-generator", "repetitions": repetitions},
    }


def test_stream_reports_saved_progress_and_correct_property_totals(client, monkeypatch):
    answers = iter(['{"budget_eur":2800000}', '{"budget_eur":2400000}', "Not valid JSON"])
    monkeypatch.setattr(
        "traceability_lab.pipeline.generate",
        lambda *args: {"choices": [{"message": {"content": next(answers)}, "finish_reason": "stop"}]},
    )
    response = client.post("/api/experiment-definitions/stream", json=request_data(client))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines()]
    assert [event["type"] for event in events] == ["started", "progress", "progress", "progress", "completed"]
    assert [event["completed"] for event in events if event["type"] == "progress"] == [1, 2, 3]
    result = events[-1]
    assert len(result["run_ids"]) == 3
    assert result["summary"]["planned_total"] == 3
    assert result["summary"]["experiment_title"] == "Progress experiment"
    summary = client.get(f"/api/batches/{result['batch_id']}").json()
    assert result["summary"] == summary
    assert [row["answer"] for row in summary["runs"]] == [
        '{"budget_eur":2800000}',
        '{"budget_eur":2400000}',
        "Not valid JSON",
    ]
    counts = {item["id"]: item for item in summary["property_counts"]}
    assert counts["budget_eur"] == {
        "id": "budget_eur",
        "total": 3,
        "pass": 1,
        "fail": 1,
        "requires_review": 1,
        "not_evaluated": 0,
    }
    assert counts["answer_json"]["pass"] == 2
    for identity in result["run_ids"]:
        run = client.get(f"/api/runs/{identity}").json()
        assert "2800000" not in json.dumps(run["snapshot"]["request_body"])


def test_events_are_emitted_between_runs_not_after_the_entire_batch(client, db, settings):
    request = DefinedExperimentRun(**request_data(client, repetitions=2))
    events = prepare_definition_runs(
        db, settings, client.app.state.embedder, client.app.state.tracing, request
    )
    assert next(events)["type"] == "started"
    assert db.rows("SELECT id FROM runs") == []
    first = next(events)
    assert first["completed"] == 1
    assert len(db.rows("SELECT id FROM runs")) == 1
    assert db.one("SELECT run_id FROM run_evaluations WHERE run_id=?", (first["run_id"],))
    assert next(events)["completed"] == 2
    assert next(events)["type"] == "completed"


def test_invalid_selection_is_rejected_before_streaming(client):
    request = request_data(client)
    request["definition"]["version_ids"] = ["missing"]
    response = client.post("/api/experiment-definitions/stream", json=request)
    assert response.status_code == 422
    assert client.get("/api/runs").json() == []


def test_failed_model_calls_are_counted_as_attempts_and_never_pass(client, monkeypatch):
    calls = 0

    def generate(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("Controlled test failure")
        return {"choices": [{"message": {"content": '{"budget_eur":2800000}'}, "finish_reason": "stop"}]}

    monkeypatch.setattr("traceability_lab.pipeline.generate", generate)
    response = client.post("/api/experiment-definitions/stream", json=request_data(client, repetitions=2))
    result = json.loads(response.text.splitlines()[-1])
    assert result["type"] == "completed"
    assert result["summary"]["counts"] == {"total": 2, "completed": 1, "failed": 1}
    budget = next(item for item in result["summary"]["property_counts"] if item["id"] == "budget_eur")
    assert budget["pass"] == 1 and budget["requires_review"] == 1 and budget["total"] == 2


def test_internal_evaluation_failure_emits_error_not_false_completion(client, monkeypatch):
    def fail(*args):
        raise ValueError("Controlled evaluation failure")

    monkeypatch.setattr("traceability_lab.evaluation.persist_automatic_evaluation", fail)
    response = client.post("/api/experiment-definitions/stream", json=request_data(client))
    events = [json.loads(line) for line in response.text.splitlines()]
    assert [event["type"] for event in events] == ["started", "error"]
    assert "Controlled evaluation failure" in events[-1]["message"]
    assert len(client.get("/api/runs").json()) == 1


def test_chart_denominator_includes_unknown_and_inapplicable_results():
    result = property_counts(
        [
            {"automatic": {"criteria": [{"id": "budget", "applicability": "applicable", "result": "pass"}]}},
            {
                "automatic": {
                    "criteria": [{"id": "budget", "applicability": "undetermined", "result": "not_evaluated"}]
                }
            },
            {"automatic": None},
        ]
    )[0]
    assert result == {
        "id": "budget",
        "total": 3,
        "pass": 1,
        "fail": 0,
        "requires_review": 0,
        "not_evaluated": 2,
    }
    assert ExperimentInput(repetitions=100).repetitions == 100


def test_progress_does_not_recompute_previous_run_evaluations(client, monkeypatch):
    from traceability_lab import evaluation

    original, calls = evaluation.evaluate, []

    def counted(db, identity):
        calls.append(identity)
        return original(db, identity)

    monkeypatch.setattr(evaluation, "evaluate", counted)
    response = client.post("/api/experiment-definitions/stream", json=request_data(client, repetitions=3))
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1]["type"] == "completed"
    assert len(calls) == len(set(calls)) == 3
