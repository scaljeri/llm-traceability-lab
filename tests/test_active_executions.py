"""Background batches survive disconnected observers and cancel active model requests."""

import threading
import time

from test_defined_experiments import definition


def wait_for(client, identity, status):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = client.get(f"/api/executions/{identity}").json()
        if result.get("status") == status:
            return result
        time.sleep(0.01)
    raise AssertionError(result)


def test_cancel_interrupts_current_run_and_keeps_only_completed_results(client, monkeypatch):
    from traceability_lab.cancellation import cancellation_event, check_cancelled
    from traceability_lab.llm import generate

    entered = threading.Event()
    calls = []

    def blocked(settings, body, chunks):
        calls.append(body)
        if len(calls) == 2:
            entered.set()
            assert cancellation_event().wait(3)
            check_cancelled()
        return generate(settings, body, chunks)

    monkeypatch.setattr("traceability_lab.pipeline.generate", blocked)
    request = {"definition": definition(client), "execution": {"repetitions": 4}}
    started = client.post("/api/executions", json=request)
    assert started.status_code == 202, started.text
    identity = started.json()["batch_id"]
    assert entered.wait(3)
    active = client.get("/api/executions/active").json()
    assert active["batch_id"] == identity and active["completed"] == 1
    assert active["status"] == "running"
    assert active["summary"]["counts"]["total"] == 1
    assert len(active["summary"]["evaluations"]) == 1
    assert len(active["summary"]["runs"]) == 1
    assert client.post("/api/executions", json=request).status_code == 409
    begin = time.monotonic()
    client.post(f"/api/executions/{identity}/cancel")
    final = wait_for(client, identity, "cancelled")
    assert time.monotonic() - begin < 1
    assert final["completed"] == 1
    assert final["summary"]["counts"]["total"] == 1
    assert len(calls) == 2
    assert client.get(f"/api/batches/{identity}").json()["counts"]["total"] == 1
    records = [run for run in client.get("/api/runs").json() if run["batch_id"] == identity]
    assert len(records) == 2
    interrupted = next(run for run in records if run["status"] == "failed")
    log = client.get(f"/api/runs/{interrupted['id']}/evaluation").json()
    assert log["automatic"]["semantic_evaluation"]["status"] == "cancelled"
    assert client.get("/api/executions/active").json() is None
    assert client.post(f"/api/executions/{identity}/cancel").json()["status"] == "cancelled"


def test_background_completion_keeps_results_for_reconnecting_observer(client):
    response = client.post(
        "/api/executions", json={"definition": definition(client), "execution": {"repetitions": 2}}
    )
    assert response.status_code == 202, response.text
    final = wait_for(client, response.json()["batch_id"], "completed")
    assert final["completed"] == 2
    assert final["summary"]["counts"]["total"] == 2
    assert client.get("/api/executions/missing").status_code == 404


def test_background_failure_can_be_inspected_and_does_not_block_next_batch(client, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("Generator unavailable")

    monkeypatch.setattr("traceability_lab.pipeline.run_question", fail)
    request = {"definition": definition(client), "execution": {"repetitions": 2}}
    response = client.post("/api/executions", json=request)
    final = wait_for(client, response.json()["batch_id"], "failed")
    assert "Generator unavailable" in final["error"]
    assert final["completed"] == 0
    assert final["summary"] is None
    assert client.get("/api/executions/active").json() is None
    second = client.post("/api/executions", json=request)
    assert second.status_code == 202
    wait_for(client, second.json()["batch_id"], "failed")


def test_cancelled_before_first_run_has_no_results(client):
    from threading import Event

    from traceability_lab.defined_experiments import prepare_definition_runs
    from traceability_lab.models import DefinedExperimentRun

    cancel = Event()
    cancel.set()
    app = client.app

    executions = app.state.executions
    request = DefinedExperimentRun.model_validate(
        {"definition": definition(client), "execution": {"repetitions": 5}}
    )
    events = list(
        prepare_definition_runs(
            executions.db,
            executions.settings,
            executions.embedder,
            executions.tracing,
            request,
            cancel_event=cancel,
        )
    )
    assert [event["type"] for event in events] == ["started", "cancelled"]
    assert events[-1]["summary"] is None
    assert events[-1]["run_ids"] == []


def test_cancel_during_review_stops_remaining_reviews_and_excludes_incomplete_run(client, monkeypatch):
    from test_quoted_facts import configure_calls, quoted_experiment

    from traceability_lab.cancellation import cancellation_event, check_cancelled

    configure_calls(monkeypatch)
    entered = threading.Event()
    calls = []

    def blocked(*args):
        calls.append(args)
        entered.set()
        assert cancellation_event().wait(3)
        check_cancelled()

    monkeypatch.setattr("traceability_lab.semantic_evaluation.generate", blocked)
    response = client.post(
        "/api/executions",
        json={
            "definition": quoted_experiment(),
            "execution": {"llm_profile_id": "test-generator", "repetitions": 3},
        },
    )
    identity = response.json()["batch_id"]
    assert entered.wait(3)
    client.post(f"/api/executions/{identity}/cancel")
    result = wait_for(client, identity, "cancelled")
    assert result["completed"] == 0
    assert result["summary"] is None
    assert len(calls) == 1
    runs = [run for run in client.get("/api/runs").json() if run["batch_id"] == identity]
    assert len(runs) == 1 and runs[0]["status"] == "completed"  # Generation completed; review was cancelled.
    review = client.get(f"/api/runs/{runs[0]['id']}/evaluation").json()["automatic"]["semantic_evaluation"]
    assert review["status"] == "cancelled"
    assert review["calls"][0]["status"] == "cancelled"
    assert client.get(f"/api/batches/{identity}").status_code == 404


def test_unchanged_poll_omits_summary_but_terminal_state_keeps_it():
    from traceability_lab.active_executions import ActiveExecutions

    executions = ActiveExecutions(None, None, None, None)
    executions.jobs["example"] = {
        "state": {"status": "running", "completed": 1, "summary": {"large": "saved result"}}
    }
    assert executions.get("example", known_completed=1)["summary"] is None
    assert executions.get("example", known_completed=0)["summary"] == {"large": "saved result"}
    assert executions.jobs["example"]["state"]["summary"] == {"large": "saved result"}
    executions.jobs["example"]["state"]["status"] = "completed"
    assert executions.get("example", known_completed=1)["summary"] == {"large": "saved result"}
