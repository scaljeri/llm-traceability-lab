"""De hele pipeline testen met expliciet gelabelde, deterministische testdoubles."""

import json

import pytest
from prov.model import ProvDocument


def test_run_records_inputs_ranked_retrievals_and_real_otel_spans(client, completed, db):
    run = completed
    assert run["status"] == "completed", run["error"]
    assert run["backend"] == "demo"
    assert run["answer"].startswith("DEMO OUTPUT")
    assert run["snapshot"]["request_body"]["messages"] == run["messages"]
    assert len(run["snapshot"]["query_vector"]) == 256
    distances = [r["distance"] for r in run["retrievals"]]
    assert distances == sorted(distances)
    assert [r["rank"] for r in run["retrievals"]] == list(range(1, 5))
    for result in run["retrievals"]:
        assert result["chunk_id"] in run["messages"][1]["content"]
        assert result["text"] in run["messages"][1]["content"]
    spans = client.get(f"/api/runs/{run['id']}/trace").json()
    assert set(run["snapshot"]["required_spans"]) <= {span["name"] for span in spans}
    root = next(span for span in spans if span["name"] == "rag.request")
    assert root["parent_id"] is None
    assert all(span["trace_id"] == run["trace_id"] for span in spans)
    assert all(span["parent_id"] == root["span_id"] for span in spans if span != root)
    assert all(span["duration_ms"] >= 0 for span in spans)
    assert run["question"] not in json.dumps(spans)
    assert db.one("SELECT COUNT(*) AS n FROM embeddings")["n"] == 5


def test_selected_old_version_does_not_leak_new_version(client):
    batch = client.post("/test-scenarios/version-v1/run", json={}).json()
    run = client.get(f"/api/runs/{batch['run_ids'][0]}").json()
    assert [result["version"] for result in run["retrievals"]] == ["1"]
    assert "2,4 miljoen" in run["answer"]
    assert "2,8 miljoen" not in run["messages"][1]["content"]
    assert len(run["snapshot"]["corpus"]) == 1


def test_exact_query_can_be_replayed_without_changing_old_run(client, completed):
    before = client.get(f"/api/runs/{completed['id']}/report").json()
    versions = [item["id"] for item in completed["snapshot"]["corpus"]]
    client.post(
        "/api/documents",
        json={
            "document_id": "orion",
            "title": "Orion",
            "version": "3",
            "source": "test",
            "content": "Orion begon in 2099.",
        },
    )
    replay = client.post(
        "/api/questions", json={"question": completed["question"], "version_ids": versions}
    ).json()
    assert replay["id"] != completed["id"]
    assert replay["trace_id"] != completed["trace_id"]
    assert replay["messages"] == completed["messages"]
    assert replay["retrievals"] == [{**row, "run_id": replay["id"]} for row in completed["retrievals"]]
    assert client.get(f"/api/runs/{completed['id']}/report").json() == before


def test_failed_model_attempt_retains_context_and_no_false_success(client, monkeypatch):
    def fail(*_args):
        raise ConnectionError("lokale modelserver niet bereikbaar")

    monkeypatch.setattr("traceability_lab.pipeline.generate", fail)
    result = client.post("/api/questions", json={"question": "Wanneer startte Orion?"})
    run = result.json()
    assert result.status_code == 201
    assert run["status"] == "failed"
    assert run["answer"] is None
    assert run["messages"] and run["retrievals"]
    spans = client.get(f"/api/runs/{run['id']}/trace").json()
    assert next(span for span in spans if span["name"] == "llm.generate")["status"] == "ERROR"
    assert "lokale modelserver" not in json.dumps(spans)
    metrics = client.get(f"/api/runs/{run['id']}/evaluation").json()
    assert "run.persist" in metrics["missing_trace_stages"]


def test_prov_json_roundtrips_and_does_not_assert_claim_support(client, completed):
    payload = client.get(f"/api/runs/{completed['id']}/provenance").json()
    assert '"None"' not in json.dumps(payload)
    parsed = ProvDocument.deserialize(content=json.dumps(payload), format="json")
    assert parsed.get_records()
    assert "used" in payload and "wasGeneratedBy" in payload and "wasDerivedFrom" in payload
    assert "supported" not in json.dumps(payload)
    assert any(identity.startswith("lab:embedding-") for identity in payload["entity"])
    inference = next(
        item for item in payload["activity"].values() if item.get("prov:label") == "llm.generate"
    )
    assert inference["lab:span_id"]
    provn = client.get(f"/api/runs/{completed['id']}/provenance?format=provn")
    assert provn.status_code == 200
    assert provn.text.startswith("document") and "endDocument" in provn.text
    assert '"None"' not in provn.text


def test_report_replays_retrieval_without_source_database_or_models(client, completed):
    from traceability_lab.replay import replay_retrieval

    report = client.get(f"/api/runs/{completed['id']}/report").json()
    assert report["evaluation_reference"]["expected"] == "12 maart 2024"
    assert len(report["embeddings"]) == 5
    assert replay_retrieval(report)["matches_saved_retrieval"]
    report["run"]["retrievals"][0]["distance"] += 0.1
    assert not replay_retrieval(report)["matches_saved_retrieval"]


def test_finished_run_cannot_be_rewritten(client, completed, db):
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError, match="immutable completed run"):
        with db.connect() as connection:
            connection.execute("UPDATE runs SET answer='changed' WHERE id=?", (completed["id"],))


@pytest.mark.parametrize("case_id", ["complete", "missing", "conflict", "version", "version-v1", "knowledge"])
def test_all_experiments_run_without_ground_truth_in_prompt(client, case_id):
    response = client.post(f"/test-scenarios/{case_id}/run", json={"repetitions": 2})
    assert response.status_code == 201, response.text
    batch = client.get(f"/api/batches/{response.json()['batch_id']}").json()
    assert batch["counts"] == {"total": 2, "completed": 2, "failed": 0}
    for run_id in response.json()["run_ids"]:
        run = client.get(f"/api/runs/{run_id}").json()
        assert run["mode"] == "experiment"
        assert "required_evidence" not in json.dumps(run["snapshot"]["request_body"])
        assert "expected" not in json.dumps(run["snapshot"]["request_body"])
        assert "Parijs" not in json.dumps(run["snapshot"]["request_body"])
