"""Deleted labels can be reused without changing historical source identities."""

from fastapi.testclient import TestClient
from legacy_api import legacy_app as create_app


def test_recreate_deleted_pair_keeps_old_runs_and_is_idempotent(client, completed):
    old_run = client.get(f"/api/runs/{completed['id']}").json()
    original = next(
        doc
        for doc in client.get("/api/documents").json()
        if doc["document_id"] == "orion" and doc["version"] == "2"
    )
    assert client.delete(f"/api/document-versions/{original['id']}").status_code == 200
    data = {
        "document_id": "orion",
        "version": "2",
        "filename": "orion-v2.txt",
        "body": "New replacement content.",
    }
    created = client.post("/api/documents", json=data)
    assert created.status_code == 201, created.text
    saved = created.json()
    assert saved["id"] != original["id"]
    assert saved["indexing"]["status"] == "ready"
    assert client.post("/api/documents", json=data).json()["id"] == saved["id"]
    active = [
        doc
        for doc in client.get("/api/documents").json()
        if doc["document_id"] == "orion" and doc["version"] == "2"
    ]
    assert len(active) == 1
    assert client.get(f"/api/runs/{completed['id']}").json() == old_run
    report = client.get(f"/api/runs/{completed['id']}/report").json()
    assert next(doc for doc in report["documents"] if doc["id"] == original["id"])["body"] == original["body"]
    run = client.post("/test-scenarios/complete/run", json={}).json()
    latest = client.get(f"/api/runs/{run['run_ids'][0]}").json()
    assert saved["id"] in [doc["id"] for doc in latest["snapshot"]["corpus"]]
    client.delete(f"/api/document-versions/{saved['id']}")
    newer = client.post("/api/documents", json={**data, "body": "A second replacement."})
    assert newer.status_code == 201
    assert newer.json()["id"] not in {saved["id"], original["id"]}


def test_schema_four_migration_removes_full_unique_constraint_and_preserves_history(
    client, db, settings, completed
):
    original_run = client.get(f"/api/runs/{completed['id']}").json()
    original = next(
        doc
        for doc in client.get("/api/documents").json()
        if doc["document_id"] == "orion" and doc["version"] == "1"
    )
    client.delete(f"/api/document-versions/{original['id']}")
    with db.connect() as connection:
        connection.execute("CREATE UNIQUE INDEX legacy_pair ON document_versions(document_id,version)")
        connection.execute("PRAGMA user_version=4")
    with TestClient(create_app(settings)) as restarted:
        response = restarted.post(
            "/api/documents",
            json={
                "document_id": "orion",
                "version": "1",
                "filename": "fresh.txt",
                "body": "Recreated after migration.",
            },
        )
        assert response.status_code == 201, response.text
        assert restarted.get(f"/api/runs/{completed['id']}").json() == original_run
    with TestClient(create_app(settings)) as restarted:
        active = [
            doc
            for doc in restarted.get("/api/documents").json()
            if doc["document_id"] == "orion" and doc["version"] == "1"
        ]
        assert len(active) == 1
        assert active[0]["body"] == "Recreated after migration."
    with db.connect() as connection:
        assert not connection.execute("PRAGMA foreign_key_check").fetchall()
