"""Embedding-identiteiten, exacte vectorsearch en ontbrekende lokale modellen."""

from fastapi.testclient import TestClient
from legacy_api import legacy_app as create_app

from traceability_lab.config import Settings
from traceability_lab.embeddings import Embedder, artifact_hash
from traceability_lab.retrieval import ensure_embeddings, retrieve


def test_local_artifact_fingerprint_is_order_stable_and_tracks_bytes(tmp_path):
    # Ook een model dat onder een .cache-map staat moet gehasht kunnen worden.
    directory = tmp_path / ".cache" / "snapshot"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    (directory / "weights.bin").write_bytes(b"weights")
    before = artifact_hash(directory)
    assert artifact_hash(directory) == before
    (directory / "weights.bin").write_bytes(b"different weights")
    assert artifact_hash(directory) != before


def test_retrieval_uses_only_same_embedding_configuration(client, db, settings):
    from traceability_lab.documents import version_chunks

    versions = client.get("/api/documents").json()
    chunks = version_chunks(db, [item["id"] for item in versions])
    embedder = Embedder(settings)
    ensure_embeddings(db, chunks, embedder)
    original_count = db.one("SELECT COUNT(*) n FROM embeddings")["n"]
    ensure_embeddings(db, chunks, embedder)
    assert db.one("SELECT COUNT(*) n FROM embeddings")["n"] == original_count
    vector = embedder.encode(f"Document ID: {chunks[0]['document_id']}\n\n{chunks[0]['text']}")
    found = retrieve(db, chunks, vector, embedder.key(), 1)
    assert found[0]["id"] == chunks[0]["id"]
    assert abs(found[0]["distance"]) < 1e-5
    assert retrieve(db, chunks, vector, "unrelated-model", 1) == []


def test_local_mode_fails_explicitly_without_embedding_model(tmp_path):
    settings = Settings(
        backend="local",
        database=tmp_path / "local.sqlite3",
        embedding_path=None,
        otlp_endpoint=None,
    )
    with TestClient(create_app(settings)) as client:
        from scenario_fixtures import bootstrap

        from traceability_lab.db import Database

        bootstrap(Database(settings.database))
        run = client.post("/api/questions", json={"question": "Wanneer begon Orion?"}).json()
        assert run["status"] == "failed"
        assert run["backend"] == "local"
        assert run["answer"] is None
        assert "Settings.embedding_path" in run["error"]
        assert run["messages"] == []
