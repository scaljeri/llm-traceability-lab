"""Editing preserves originals and explicitly reports embedding failures."""

import json

import httpx


def test_edit_creates_revision_and_rebuilds_vectors(client, db, completed, monkeypatch):
    original = next(
        doc
        for doc in client.get("/api/documents").json()
        if doc["document_id"] == "orion" and doc["version"] == "2"
    )
    before = client.get(f"/api/runs/{completed['id']}").json()
    calls = []
    original_client = httpx.Client

    def handler(request):
        assert request.url.path == "/v1/embeddings"
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json={"data": [{"embedding": [1.0] + [0.0] * 1023}]})

    monkeypatch.setattr(
        "httpx.Client", lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs)
    )
    edited = client.post(
        f"/api/documents/{original['id']}/revisions",
        json={
            "title": original["title"],
            "content": "Updated budget: 4 million.",
            "embedding_profile_id": "test-embedding",
        },
    )
    assert edited.status_code == 201, edited.text
    saved = edited.json()
    assert saved["version"] == "3"
    assert saved["previous_version_id"] == original["id"]
    assert saved["sha256"] != original["sha256"]
    assert saved["indexing"]["status"] == "ready"
    assert calls[0]["input"] == "Document ID: orion\n\nUpdated budget: 4 million."
    assert calls[0]["model"] == "bge-m3"
    assert (
        db.one("SELECT content FROM source_versions WHERE id=?", (original["id"],))["content"]
        == original["content"]
    )
    assert client.get(f"/api/runs/{completed['id']}").json() == before
    chunks = client.get(f"/api/documents/{saved['id']}/chunks").json()
    assert db.one("SELECT COUNT(*) n FROM embeddings WHERE chunk_id=?", (chunks[0]["id"],))["n"] == 1


def test_save_failure_keeps_revision_and_retry_does_not_create_another(client, monkeypatch):
    version = client.get("/api/documents").json()[0]
    original = __import__("legacy_api", fromlist=["ensure_embeddings"]).ensure_embeddings

    def fail(*args):
        raise ValueError("embedding server unavailable")

    monkeypatch.setattr("legacy_api.ensure_embeddings", fail)
    response = client.post(
        f"/api/documents/{version['id']}/revisions",
        json={"title": version["title"], "content": "New content"},
    )
    assert response.status_code == 201
    saved = response.json()
    assert saved["indexing"]["status"] == "failed"
    assert "unavailable" in saved["indexing"]["error"]
    count = len(client.get("/api/documents").json())
    monkeypatch.setattr("legacy_api.ensure_embeddings", original)
    retried = client.post(f"/api/documents/{saved['id']}/embeddings", json={})
    assert retried.json()["status"] == "ready"
    assert len(client.get("/api/documents").json()) == count


def test_changing_document_id_reembeds_same_body_and_preserves_original(client, db, monkeypatch):
    original = next(doc for doc in client.get("/api/documents").json() if doc["document_id"] == "orion")
    calls = []
    original_client = httpx.Client

    def handler(request):
        calls.append(json.loads(request.content)["input"])
        return httpx.Response(200, json={"data": [{"embedding": [1.0] + [0.0] * 1023}]})

    monkeypatch.setattr(
        "httpx.Client", lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs)
    )
    saved = client.post(
        f"/api/documents/{original['id']}/revisions",
        json={
            "document_id": "renamed-orion",
            "version_number": 1,
            "title": original["title"],
            "content": original["body"],
            "embedding_profile_id": "test-embedding",
        },
    )
    assert saved.status_code == 201, saved.text
    edited = saved.json()
    assert edited["document_id"] == "renamed-orion"
    assert edited["sha256"] == original["sha256"]
    assert edited["id"] != original["id"]
    assert edited["indexing"]["status"] == "ready"
    assert calls == [f"Document ID: renamed-orion\n\n{original['body']}"]
    assert edited["indexing"]["model"]["document_input_format"] == "document-id-and-chunk-v1"
    assert (
        db.one("SELECT document_id FROM document_versions WHERE id=?", (original["id"],))["document_id"]
        == "orion"
    )


def test_edit_updates_same_visible_version_and_preserves_historical_report(client, completed, settings):
    documents = client.get("/api/documents").json()
    original = next(doc for doc in documents if doc["document_id"] == "orion" and doc["version"] == "2")
    report_path = f"/api/runs/{completed['id']}/report"
    before = client.get(report_path).json()
    response = client.put(
        f"/api/documents/{original['id']}",
        json={
            "title": original["title"],
            "filename": original["filename"],
            "content": "Edited in place: 4 million.",
        },
    )
    assert response.status_code == 200, response.text
    saved = response.json()
    current = client.get("/api/documents").json()
    assert len(current) == len(documents)
    assert saved["version"] == original["version"] == "2"
    assert saved["filename"] == original["filename"]
    assert saved["id"] != original["id"]
    assert [doc["body"] for doc in current if doc["document_id"] == "orion" and doc["version"] == "2"] == [
        "Edited in place: 4 million."
    ]
    assert saved["indexing"]["status"] == "ready"
    assert (
        client.get(f"/api/documents/{saved['id']}/chunks").json()[0]["text"] == "Edited in place: 4 million."
    )
    after = client.get(report_path).json()
    assert {key: value for key, value in after.items() if key != "documents"} == {
        key: value for key, value in before.items() if key != "documents"
    }
    # Retirement state is current catalog metadata; source bytes and all run artifacts stay exact.
    assert [
        {key: value for key, value in doc.items() if key != "deleted_at"} for doc in after["documents"]
    ] == [{key: value for key, value in doc.items() if key != "deleted_at"} for doc in before["documents"]]
    stale = client.put(
        f"/api/documents/{original['id']}", json={"title": original["title"], "content": "Stale editor"}
    )
    assert stale.status_code == 422
    assert client.get("/api/documents").json() == current
    from fastapi.testclient import TestClient
    from legacy_api import legacy_app as create_app

    with TestClient(create_app(settings)) as restarted:
        assert restarted.get("/api/documents").json() == current


def test_unchanged_edit_is_idempotent(client):
    original = client.get("/api/documents").json()[0]
    response = client.put(
        f"/api/documents/{original['id']}",
        json={"title": original["title"], "filename": original["filename"], "content": original["body"]},
    )
    assert response.status_code == 200, response.text
    assert response.json()["id"] == original["id"]


def test_edit_cannot_overwrite_another_document_or_change_version(client):
    before = client.get("/api/documents").json()
    original = next(doc for doc in before if doc["document_id"] == "orion" and doc["version"] == "1")
    for extra in [{"document_id": "finance"}, {"version_number": 2}]:
        response = client.put(
            f"/api/documents/{original['id']}",
            json={"title": original["title"], "content": "Changed", **extra},
        )
        assert response.status_code == 422, response.text
        assert client.get("/api/documents").json() == before


def test_failed_replacement_rolls_back_archiving(client):
    original = client.post(
        "/api/documents",
        json={"document_id": "atomic-old", "version": "1", "content": "Original", "sequence_number": 1},
    ).json()
    client.post(
        "/api/documents",
        json={"document_id": "atomic-target", "version": "2", "content": "Other", "sequence_number": 1},
    )
    before = client.get("/api/documents").json()
    response = client.put(
        f"/api/documents/{original['id']}",
        json={"document_id": "atomic-target", "title": original["title"], "content": "Collision"},
    )
    assert response.status_code == 409
    assert client.get("/api/documents").json() == before


def test_edit_survives_embedding_failure_and_can_retry_without_extra_version(client, monkeypatch):
    import legacy_api as main

    original = client.get("/api/documents").json()[0]
    before_count = len(client.get("/api/documents").json())
    ensure = main.ensure_embeddings

    def fail(*args):
        raise ValueError("Test indexing failure")

    monkeypatch.setattr(main, "ensure_embeddings", fail)
    response = client.put(
        f"/api/documents/{original['id']}",
        json={"title": original["title"], "content": "Saved edited content"},
    )
    assert response.status_code == 200
    saved = response.json()
    assert saved["indexing"]["status"] == "failed"
    assert saved["version"] == original["version"]
    assert len(client.get("/api/documents").json()) == before_count
    assert (
        next(doc for doc in client.get("/api/documents").json() if doc["id"] == saved["id"])["body"]
        == "Saved edited content"
    )
    monkeypatch.setattr(main, "ensure_embeddings", ensure)
    assert client.post(f"/api/documents/{saved['id']}/embeddings", json={}).json()["status"] == "ready"
    assert len(client.get("/api/documents").json()) == before_count
