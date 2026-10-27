"""Bronintegriteit en versiebehoud zijn deterministisch te bewijzen."""

import hashlib
import sqlite3

import pytest

from traceability_lab.documents import chunk_text


def test_chunk_offsets_reconstruct_unicode_and_whitespace():
    source = "  Héél precies 👩🏽‍🔬\n\nGeen normalisatie.\r\nLaatste regel  "
    chunks = chunk_text(source, size=7)
    assert "".join(chunk["text"] for chunk in chunks) == source
    assert chunks == chunk_text(source, size=7)
    for chunk in chunks:
        assert source[chunk["start"] : chunk["end"]] == chunk["text"]
        assert hashlib.sha256(chunk["text"].encode()).hexdigest() == chunk["sha256"]
    with pytest.raises(ValueError):
        chunk_text(source, 0)


def test_import_is_idempotent_and_different_content_requires_new_version(client, db):
    document = {
        "document_id": "test",
        "title": "Test",
        "source": "local/test.txt",
        "version": "1",
        "content": "Eerste inhoud.",
    }
    first = client.post("/api/documents", json=document)
    assert first.status_code == 201
    assert client.post("/api/documents", json=document).json()["id"] == first.json()["id"]
    document["content"] = "Gewijzigd."
    assert client.post("/api/documents", json=document).status_code == 422
    document["version"] = "2"
    second = client.post("/api/documents", json=document)
    assert second.status_code == 201
    assert second.json()["id"] != first.json()["id"]
    assert (
        db.one("SELECT content FROM source_versions WHERE id=?", (first.json()["id"],))["content"]
        == "Eerste inhoud."
    )


def test_database_rejects_source_rewrites_and_broken_foreign_keys(client, db):
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with db.connect() as connection:
            connection.execute("UPDATE document_versions SET body='corrupt'")
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        with db.connect() as connection:
            connection.execute("INSERT INTO embeddings VALUES ('missing','model',x'00000000',1,'{}')")


def test_new_installation_starts_without_demo_documents(settings):
    from fastapi.testclient import TestClient

    from traceability_lab.main import create_app

    for _ in range(2):
        with TestClient(create_app(settings)) as client:
            assert client.get("/api/documents").status_code == 404
            assert client.get("/api/experiments").status_code == 404
            assert client.post("/api/experiments/complete/run", json={}).status_code == 404
