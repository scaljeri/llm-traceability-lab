"""One document-version row stores filename, timestamp, content hash and body."""

from datetime import datetime
from hashlib import sha256


def test_flat_document_fields_and_no_logical_document_tables(client, db):
    body = "Orion budget: €3 million."
    result = client.post(
        "/api/documents",
        json={"document_id": "simple", "version": "2", "filename": "orion-v2.txt", "body": body},
    )
    assert result.status_code == 201, result.text
    saved = result.json()
    assert saved["filename"] == "orion-v2.txt"
    assert saved["body"] == body
    assert datetime.fromisoformat(saved["created_at"]).tzinfo is not None
    assert saved["sha256"] == sha256(body.encode()).hexdigest()
    assert db.one("SELECT body FROM document_versions WHERE id=?", (saved["id"],))["body"] == body
    assert not db.rows(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('documents','document_catalog')"
    )
    original = client.get("/api/documents").json()
    response = client.post(
        "/api/documents",
        json={"document_id": "simple", "version": "1", "filename": "orion-v1.txt", "body": "Earlier text"},
    )
    assert response.status_code == 201
    assert len(client.get("/api/documents").json()) == len(original) + 1
    assert client.delete(f"/api/document-versions/{response.json()['id']}").status_code == 200
    assert any(doc["id"] == saved["id"] for doc in client.get("/api/documents").json())
