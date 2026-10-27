"""Portable version labels, atomic imports and experiment ordering across databases."""

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from traceability_lab.config import Settings
from traceability_lab.documents import digest
from traceability_lab.main import create_app


def experiment(client):
    docs = client.get("/api/documents").json()
    ids = [
        next(doc["id"] for doc in docs if doc["document_id"] == "orion" and doc["version"] == version)
        for version in ["2", "1"]
    ]
    result = client.post(
        "/api/experiment-definitions",
        json={
            "title": "Portable experiment",
            "question": "A research question?",
            "version_ids": ids,
            "system_prompt": "Policy",
            "context": "{{user_question}}\n{{documents}}\n{{output_json}}",
            "answer_json": {"value": "string | null"},
            "validation_json": {"value": "expected"},
        },
    )
    assert result.status_code == 201
    return result.json()


def test_documents_export_is_array_with_chosen_ids_and_roundtrips(client):
    exported = client.get("/api/documents/export").json()
    assert isinstance(exported, list)
    assert all(row["id"] == row["document_id"] + "@" + row["version"] for row in exported)
    assert all(digest(row["content"]) == row["sha256"] for row in exported)
    before = client.get("/api/documents").json()
    imported = client.post("/api/documents/import", json=exported)
    assert imported.status_code == 200, imported.text
    assert imported.json()["created"] == []
    assert len(imported.json()["unchanged"]) == len(exported)
    assert client.get("/api/documents").json() == before


def test_document_import_is_atomic_and_never_overwrites(client):
    exported = client.get("/api/documents/export").json()
    conflict = deepcopy(exported[0])
    conflict["content"] += " DIFFERENT"
    conflict["sha256"] = digest(conflict["content"])
    response = client.post(
        "/api/documents/import",
        json=[
            {"document_id": "new-stable-name", "version": 1, "content": "Fresh text"},
            conflict,
        ],
    )
    assert response.status_code == 422
    assert "Nothing imported" in response.text
    assert client.get("/api/documents/export").json() == exported


@pytest.mark.parametrize("change", ["hash", "id", "duplicate", "invalid"])
def test_malformed_document_import_does_not_write(client, change):
    exported = client.get("/api/documents/export").json()
    data = deepcopy(exported[:1])
    if change == "hash":
        data[0]["sha256"] = "0" * 64
    elif change == "id":
        data[0]["id"] = "database-random-id"
    elif change == "duplicate":
        data += data
    else:
        data.append({"document_id": "bad id", "version": "1", "content": "Text"})
    assert client.post("/api/documents/import", json=data).status_code == 422
    assert client.get("/api/documents/export").json() == exported


def test_experiment_export_embeds_ordered_document_copies(client):
    saved = experiment(client)
    exported = client.get(f"/api/experiment-definitions/{saved['id']}/export").json()
    assert exported["format"] == "traceability-experiment-v2"
    assert [doc["id"] for doc in exported["experiment"]["documents"]] == ["orion@2", "orion@1"]
    assert all(doc["content"] and doc["enabled"] for doc in exported["experiment"]["documents"])
    assert "version_ids" not in exported["experiment"]
    imported = client.post("/api/experiment-definitions/import", json=exported)
    assert imported.status_code == 200, imported.text
    assert imported.json()["definition"]["version_ids"] == []
    assert len(client.get("/api/experiment-definitions").json()) == 1


def test_cross_database_transfer_is_self_contained(client, tmp_path):
    saved = experiment(client)
    exported = client.get(f"/api/experiment-definitions/{saved['id']}/export").json()
    settings = Settings(backend="demo", database=tmp_path / "fresh.sqlite3", otlp_endpoint=None)
    with TestClient(create_app(settings)) as other:
        assert other.get("/api/documents").status_code == 404
        draft = other.post("/api/experiment-definitions/import", json=exported)
        assert draft.status_code == 200, draft.text
        restored = other.post("/api/experiment-definitions", json=draft.json()["definition"])
        assert restored.status_code == 201
        again = other.get(f"/api/experiment-definitions/{restored.json()['id']}/export").json()
        assert again["experiment"] == exported["experiment"]
        assert again["format"] == exported["format"]
        assert set(again["source_ids"]) == set(exported["source_ids"])
        assert all(again["source_ids"].values())
        # Internal chunk identities can change when importing into an empty database.
        assert draft.json()["warnings"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "hash", "database_ids"])
def test_bad_experiment_references_are_rejected_without_saving(client, change):
    saved = experiment(client)
    legacy = dict(saved["definition"])
    legacy.pop("version_ids")
    legacy["document_ids"] = ["orion@2", "orion@1"]
    data = {"format": "traceability-experiment-v1", "experiment": legacy, "document_hashes": {}}
    if change == "missing":
        data["experiment"]["document_ids"][0] = "absent@1"
    elif change == "duplicate":
        data["experiment"]["document_ids"] *= 2
    elif change == "hash":
        data["document_hashes"]["orion@2"] = "0" * 64
    else:
        data["experiment"]["version_ids"] = saved["definition"]["version_ids"]
    assert client.post("/api/experiment-definitions/import", json=data).status_code == 422
    assert len(client.get("/api/experiment-definitions").json()) == 1


def test_fact_mode_and_semantic_reference_transfer_without_generator_calls(client, monkeypatch):
    saved = experiment(client)
    data = client.get(f"/api/experiment-definitions/{saved['id']}/export").json()
    data["experiment"].update(
        output_mode="facts_and_answer",
        validation_json={},
        semantic_reference={"facts": ["Expected fact"], "answer": None},
        semantic_evaluator={"llm_profile_id": "test-reviewer"},
    )
    monkeypatch.setattr(
        "traceability_lab.pipeline.generate", lambda *args: pytest.fail("Import must not generate")
    )
    response = client.post("/api/experiment-definitions/import", json=data)
    assert response.status_code == 200
    assert response.json()["definition"]["semantic_reference"] == data["experiment"]["semantic_reference"]
    assert response.json()["definition"]["semantic_evaluator"]["llm_profile_id"] == "test-reviewer"


def test_portable_label_encodes_non_numeric_version(client):
    response = client.post(
        "/api/documents/import",
        json=[{"document_id": "chosen-name", "version": "draft@2 / Ω", "content": "Text"}],
    )
    assert response.status_code == 200
    records = client.get("/api/documents/export").json()
    assert (
        next(row for row in records if row["document_id"] == "chosen-name")["id"]
        == "chosen-name@draft%402%20%2F%20%CE%A9"
    )
