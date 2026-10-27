"""Controleer bewijsgrenzen, append-only beoordelingen en eerlijke noemers."""

import json


def add_first_claim(client, run):
    needle = "Project Orion is gestart op 12 maart 2024."
    start = run["answer"].index(needle)
    response = client.post(
        f"/api/runs/{run['id']}/claims",
        json={"start": start, "end": start + len(needle), "annotator": "onderzoeker"},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_annotation_validates_spans_and_adds_manual_trace(client, completed):
    claim = add_first_claim(client, completed)
    assert claim["text"] == "Project Orion is gestart op 12 maart 2024."
    assert (
        client.post(
            f"/api/runs/{completed['id']}/claims", json={"start": 0, "end": 999999, "annotator": "x"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"/api/runs/{completed['id']}/claims",
            json={"start": claim["start"], "end": claim["end"], "annotator": "x"},
        ).status_code
        == 409
    )
    spans = client.get(f"/api/runs/{completed['id']}/trace").json()
    annotation = next(span for span in spans if span["name"] == "claims.annotate")
    assert annotation["trace_id"] == completed["trace_id"]
    assert annotation["attributes"]["lab.method"] == "manual"


def test_evidence_is_exact_append_only_and_scoped(client, completed, db):
    claim = add_first_claim(client, completed)
    source = next(r for r in completed["retrievals"] if r["document_id"] == "orion")
    start = source["text"].index(claim["text"])
    body = {
        "classification": "supported",
        "scope": "retrieved",
        "chunk_id": source["chunk_id"],
        "start": start,
        "end": start + len(claim["text"]),
        "annotator": "human",
        "note": "De datum staat letterlijk in de bron.",
    }
    response = client.post(f"/api/claims/{claim['id']}/assessments", json=body)
    assert response.status_code == 201, response.text
    assert response.json()["quote"] == claim["text"]
    assert response.json()["method"] == "manual"
    assert (
        client.post(f"/api/claims/{claim['id']}/assessments", json={**body, "end": 99999}).status_code == 422
    )
    old_version = db.one("SELECT id FROM document_versions WHERE document_id='orion' AND version='1'")
    old_chunk = db.one("SELECT id FROM chunks WHERE version_id=?", (old_version["id"],))
    assert (
        client.post(
            f"/api/claims/{claim['id']}/assessments",
            json={**body, "chunk_id": old_chunk["id"], "scope": "corpus"},
        ).status_code
        == 422
    )
    revised = {
        "classification": "insufficient_evidence",
        "scope": "retrieved",
        "annotator": "second",
        "note": "Tweede beoordeling voor test.",
    }
    assert client.post(f"/api/claims/{claim['id']}/assessments", json=revised).status_code == 201
    run = client.get(f"/api/runs/{completed['id']}").json()
    assert len(run["claims"][0]["assessments"]) == 2
    provenance = client.get(f"/api/runs/{completed['id']}/provenance").json()
    assert "lab:evidence-" in json.dumps(provenance)


def test_evaluation_skips_historical_annotations(client, completed, db, monkeypatch):
    from traceability_lab.evaluation import batch_summary, evaluate
    from traceability_lab.pipeline import read_run

    claim = add_first_claim(client, completed)
    assert (
        client.post(
            f"/api/claims/{claim['id']}/assessments",
            json={
                "classification": "insufficient_evidence",
                "annotator": "x",
                "note": "Historical evidence.",
            },
        ).status_code
        == 201
    )
    assert (
        client.post(
            f"/api/runs/{completed['id']}/reviews",
            json={
                "claims_complete": True,
                "abstained": False,
                "annotator": "x",
                "note": "Historical review.",
            },
        ).status_code
        == 201
    )
    historical = read_run(db, completed["id"])
    rows = db.rows

    def without_annotations(sql, *args, **kwargs):
        assert not any(f"from {table}" in sql.lower() for table in ("claims", "assessments", "reviews")), sql
        return rows(sql, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(db, "rows", without_annotations)
        evaluation = evaluate(db, completed["id"])
        summary = batch_summary(db, completed["batch_id"])
    assert "manual_rates" not in summary
    for key in (
        "manual_review",
        "annotated_claim_coverage",
        "unsupported_claim_rate",
        "unsupported_rate_among_assessed_claims",
        "annotation_complete",
    ):
        assert key not in evaluation
    assert evaluation["trace_stage_coverage"]["value"] == 1
    assert evaluation["automatic"] is not None
    assert read_run(db, completed["id"])["claims"] == historical["claims"]
    assert db.rows("SELECT * FROM reviews WHERE run_id=?", (completed["id"],))


def test_nonretrieved_corpus_evidence_is_distinct(client, db):
    result = client.post("/api/questions", json={"question": "Orion", "top_k": 1}).json()
    claim = client.post(
        f"/api/runs/{result['id']}/claims", json={"start": 0, "end": 10, "annotator": "x"}
    ).json()
    retrieved_ids = {r["chunk_id"] for r in result["retrievals"]}
    eligible = {v["id"] for v in result["snapshot"]["corpus"]}
    other = next(
        c
        for c in db.rows("SELECT * FROM chunks")
        if c["id"] not in retrieved_ids and c["version_id"] in eligible
    )
    evidence = {
        "classification": "supported",
        "chunk_id": other["id"],
        "start": 0,
        "end": 10,
        "annotator": "x",
        "note": "Scopecontrole.",
    }
    assert client.post(f"/api/claims/{claim['id']}/assessments", json=evidence).status_code == 422
    assert (
        client.post(
            f"/api/claims/{claim['id']}/assessments", json={**evidence, "scope": "corpus"}
        ).status_code
        == 201
    )
    evaluation = client.get(f"/api/runs/{result['id']}/evaluation").json()
    assert "annotated_claim_coverage" not in evaluation
