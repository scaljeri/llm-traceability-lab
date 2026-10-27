"""Editable logical documents own immutable, numbered text versions."""

from fastapi.testclient import TestClient
from legacy_api import legacy_app as create_app


def test_title_and_delete_preserve_saved_runs(client, settings, completed):
    original = client.get(f"/api/runs/{completed['id']}").json()
    response = client.put("/api/documents/orion/title", json={"title": "Renamed Orion"})
    assert response.status_code == 200
    versions = [doc for doc in client.get("/api/documents").json() if doc["document_id"] == "orion"]
    assert len(versions) == 2
    assert all(doc["document_title"] == "Renamed Orion" for doc in versions)
    assert client.get(f"/api/runs/{completed['id']}").json() == original
    assert client.delete("/api/documents/orion").status_code == 200
    assert not any(doc["document_id"] == "orion" for doc in client.get("/api/documents").json())
    assert client.get(f"/api/runs/{completed['id']}/report").status_code == 200
    assert client.get(f"/api/runs/{completed['id']}").json() == original
    assert (
        client.post(
            "/api/questions",
            json={"question": "Test", "context_mode": "explicit", "version_ids": [versions[0]["id"]]},
        ).status_code
        == 422
    )
    with TestClient(create_app(settings)) as restarted:
        assert not any(doc["document_id"] == "orion" for doc in restarted.get("/api/documents").json())
        assert restarted.get(f"/api/runs/{completed['id']}/report").status_code == 200


def test_user_selects_body_version_number(client):
    original = next(
        doc
        for doc in client.get("/api/documents").json()
        if doc["document_id"] == "orion" and doc["version"] == "2"
    )
    data = {"title": "Project Orion", "content": "Another body", "version_number": 7}
    response = client.post(f"/api/documents/{original['id']}/revisions", json=data)
    assert response.status_code == 201
    assert response.json()["version"] == "7"
    assert response.json()["sequence_number"] == 7
    assert response.json()["indexing"]["status"] == "ready"
    duplicate = client.post(
        f"/api/documents/{original['id']}/revisions", json={**data, "content": "Different text"}
    )
    assert duplicate.status_code == 422
    assert (
        client.post(
            f"/api/documents/{original['id']}/revisions", json={**data, "version_number": 0}
        ).status_code
        == 422
    )
    assert client.put("/api/documents/orion/title", json={"title": "  "}).status_code == 422
