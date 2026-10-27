"""Version history and exact supplied context are deterministic system contracts."""

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient
from legacy_api import legacy_app as create_app

from traceability_lab.db import Database
from traceability_lab.llm import PROMPT_VERSION


def import_version(client, label, *, document_id="history", **extra):
    response = client.post(
        "/api/documents",
        json={
            "document_id": document_id,
            "version": label,
            "title": "Version history",
            "source": f"history/{label}.txt",
            "content": f"Budget in {label}: {label} million.",
            **extra,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_revision_history_is_validated_and_immutable(client, db):
    first = import_version(client, "draft", sequence_number=1, published_on="2026-01-01")
    second = import_version(
        client, "approved", sequence_number=2, previous_version_id=first["id"], effective_on="2026-02-01"
    )
    assert second["previous_version_id"] == first["id"]
    base = {
        "document_id": "other",
        "version": "2",
        "title": "Other",
        "source": "local",
        "content": "Text",
        "previous_version_id": first["id"],
    }
    assert client.post("/api/documents", json=base).status_code == 422
    assert (
        client.post(
            "/api/documents", json={**base, "document_id": "history", "sequence_number": 1}
        ).status_code
        == 422
    )
    assert client.post("/api/documents", json={**base, "previous_version_id": "missing"}).status_code == 422
    assert (
        client.post(
            "/api/documents", json={**base, "previous_version_id": None, "published_on": "invalid"}
        ).status_code
        == 422
    )
    for field in ("body", "previous_version_id", "sequence_number"):
        with pytest.raises(sqlite3.IntegrityError):
            with db.connect() as connection:
                connection.execute(f"UPDATE document_versions SET {field}=NULL WHERE id=?", (second["id"],))
    # Application checks are backed by a database trigger for direct inserts.
    with pytest.raises(sqlite3.IntegrityError, match="predecessor"):
        with db.connect() as connection:
            connection.execute(
                "INSERT INTO document_versions (id,document_id,version,title,display_title,filename,source_uri,body,sha256,created_at,metadata,previous_version_id) VALUES ('bad','foreign','1','x','x','x','x','x','x','x','{}',?)",
                (first["id"],),
            )


def test_explicit_context_preserves_all_versions_order_and_provenance(client, db, monkeypatch):
    first = import_version(client, "1", sequence_number=1, published_on="2026-01-01")
    second = import_version(
        client, "2", sequence_number=2, previous_version_id=first["id"], effective_on="2026-02-01"
    )
    # These must never be used in the direct-context route.
    monkeypatch.setattr(
        "traceability_lab.pipeline.ensure_embeddings", lambda *a: pytest.fail("unexpected embeddings")
    )
    monkeypatch.setattr("traceability_lab.pipeline.retrieve", lambda *a: pytest.fail("unexpected retrieval"))
    response = client.post(
        "/api/questions",
        json={
            "question": "Which budget?",
            "context_mode": "explicit",
            "version_ids": [second["id"], first["id"]],
            "top_k": 1,
            "version_instructions": True,
        },
    )
    assert response.status_code == 201, response.text
    run = response.json()
    assert run["status"] == "completed", run["error"]
    assert run["retrievals"] == []
    assert [item["version_id"] for item in run["context"]] == [second["id"], first["id"]]
    assert {item["origin"] for item in run["context"]} == {"explicit"}
    assert all(item["distance"] is None for item in run["context"])
    context = run["messages"][1]["content"]
    assert context.index("version: 2") < context.index("version: 1")
    assert "higher sequence number" in run["messages"][0]["content"]
    assert "embedding_key" not in run["snapshot"]
    assert run["snapshot"]["prompt_version"].endswith("+version-order-v1")
    report = client.get(f"/api/runs/{run['id']}/report").json()
    assert report["embeddings"] == []
    assert report["evaluation"]["trace_stage_coverage"]["value"] == 1
    assert report["evaluation"]["retrieval_recall_at_k"] is None
    assert report["evaluation"]["context_source_integrity"]["value"] == 1
    assert "prov:Revision" in json.dumps(report["provenance"])
    assert "lab:origin" in json.dumps(report["provenance"])
    assert '"None"' not in json.dumps(report["provenance"])
    sources = report["provenance"]["entity"]
    first_source, second_source = (sources[f"lab:version-{item['id']}"] for item in (first, second))
    assert first_source["lab:sequence_number"] == {"$": 1, "type": "xsd:int"}
    assert first_source["lab:published_on"] == "2026-01-01"
    assert "lab:effective_on" not in first_source
    assert second_source["lab:effective_on"] == "2026-02-01"
    assert "lab:published_on" not in second_source
    provn = client.get(f"/api/runs/{run['id']}/provenance?format=provn")
    assert provn.status_code == 200
    assert '"None"' not in provn.text
    assert "lab:sequence_number=1" in provn.text
    claim = client.post(
        f"/api/runs/{run['id']}/claims", json={"start": 0, "end": 4, "annotator": "test"}
    ).json()
    result = client.post(
        f"/api/claims/{claim['id']}/assessments",
        json={
            "classification": "supported",
            "scope": "retrieved",
            "chunk_id": run["context"][0]["chunk_id"],
            "start": 0,
            "end": 4,
            "annotator": "test",
            "note": "System test, not a factual judgment.",
        },
    )
    assert result.status_code == 201, result.text
    with pytest.raises(sqlite3.IntegrityError):
        with db.connect() as connection:
            connection.execute("DELETE FROM context_items WHERE run_id=?", (run["id"],))


def test_explicit_context_no_truncation_and_instruction_toggle(client):
    first = import_version(client, "long", content="Alpha " * 200)
    run = client.post(
        "/api/questions",
        json={"question": "Text?", "context_mode": "explicit", "version_ids": [first["id"]], "top_k": 1},
    ).json()
    assert run["status"] == "completed"
    assert len(run["context"]) == 3
    assert "".join(chunk["text"] for chunk in run["context"]) == first["content"]
    assert "higher sequence number" not in run["messages"][0]["content"]
    assert run["snapshot"]["prompt_version"] == PROMPT_VERSION
    assert (
        client.post("/api/questions", json={"question": "Test", "context_mode": "explicit"}).status_code
        == 422
    )
    assert (
        client.post(
            "/api/questions", json={"question": "Test", "version_ids": [first["id"], first["id"]]}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/questions",
            json={"question": "Test", "context_mode": "explicit", "version_ids": ["missing"]},
        ).status_code
        == 422
    )


def test_ancestor_export_does_not_supply_ancestor(client):
    first = import_version(client, "1")
    second = import_version(client, "2", previous_version_id=first["id"])
    run = client.post(
        "/api/questions",
        json={"question": "Budget?", "context_mode": "explicit", "version_ids": [second["id"]]},
    ).json()
    assert {item["version_id"] for item in run["context"]} == {second["id"]}
    provenance = client.get(f"/api/runs/{run['id']}/provenance").json()
    assert f"lab:version-{first['id']}" in provenance["entity"]
    for version in (first, second):
        source = provenance["entity"][f"lab:version-{version['id']}"]
        assert not {"lab:sequence_number", "lab:published_on", "lab:effective_on"} & source.keys()
    assert first["content"] not in run["messages"][1]["content"]


def test_real_schema_two_migration_keeps_old_runs_and_bytes(settings):
    settings.database.parent.mkdir(parents=True, exist_ok=True)
    schema = __import__("pathlib").Path("src/traceability_lab/schema.sql").read_text()
    start = schema.index("CREATE TABLE IF NOT EXISTS document_versions")
    end = schema.index("CREATE TABLE IF NOT EXISTS chunks")
    schema = (
        schema[:start]
        + """CREATE TABLE documents(id TEXT PRIMARY KEY);
    CREATE TABLE document_versions(id TEXT PRIMARY KEY,document_id TEXT REFERENCES documents(id),version TEXT,title TEXT,source TEXT,content TEXT,sha256 TEXT,imported_at TEXT,metadata TEXT);
    """
        + schema[end:]
    )
    with sqlite3.connect(settings.database) as connection:
        connection.executescript(schema)
        connection.execute("INSERT INTO documents VALUES ('old')")
        connection.execute(
            "INSERT INTO document_versions VALUES ('old-v','old','1','Old','old.txt','original','hash','2020','{}')"
        )
        snapshot = json.dumps({"corpus": [{"id": "old-v"}], "required_spans": []})
        connection.execute(
            "INSERT INTO runs (id,created_at,status,trace_id,question,mode,backend,snapshot) VALUES ('old-run','2020','completed','trace','Old?','strict','demo',?)",
            (snapshot,),
        )
        connection.execute(
            "INSERT INTO chunks VALUES ('old-chunk','old-v',0,0,8,'original','chunkhash','legacy')"
        )
        connection.execute("INSERT INTO retrievals VALUES ('old-run',1,'old-chunk',0.25,'cosine_distance')")
        connection.execute("PRAGMA user_version=2")
    with TestClient(create_app(settings)) as restarted:
        run = restarted.get("/api/runs/old-run").json()
        assert run["snapshot"] == json.loads(snapshot)
        assert run["context"][0]["text"] == "original"
        assert run["context"][0]["origin"] == "retrieved"
        old = next(doc for doc in restarted.get("/api/documents").json() if doc["id"] == "old-v")
        assert old["content"] == "original" and old["previous_version_id"] is None
    db = Database(settings.database)
    db.initialize()  # Idempotent restart.
    assert db.one("SELECT snapshot FROM runs WHERE id='old-run'")["snapshot"] == snapshot


def test_explicit_context_failure_preserves_inputs_without_embedding_server(client, monkeypatch):
    first = import_version(client, "1")

    def fail_generation(*args):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("traceability_lab.pipeline.generate", fail_generation)
    response = client.post(
        "/api/questions",
        json={
            "question": "Budget?",
            "context_mode": "explicit",
            "version_ids": [first["id"]],
            "llm_profile_id": "test-generator",
        },
    )
    assert response.status_code == 201
    run = response.json()
    assert run["status"] == "failed"
    assert "model unavailable" in run["error"]
    assert len(run["context"]) == 1 and run["messages"]
    assert run["snapshot"]["request_body"]["model"] == "test-generator-model"
    assert set(run["snapshot"]["profiles"]) == {"llm"}
    assert "embedding_key" not in run["snapshot"]


def test_context_limit_rejects_instead_of_silently_truncating(client):
    first = import_version(client, "large1", content="A" * 60_000)
    second = import_version(client, "large2", content="B" * 60_000)
    response = client.post(
        "/api/questions",
        json={"question": "Text?", "context_mode": "explicit", "version_ids": [first["id"], second["id"]]},
    )
    assert response.status_code == 422
    assert "100,000" in response.json()["detail"]
    assert client.get("/api/runs").json() == []
