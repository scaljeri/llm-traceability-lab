"""Document ownership, disabled state, order, immutable snapshots and self-contained transfer."""

import json
from copy import deepcopy

import pytest


def definition():
    return {
        "title": "Owned experiment",
        "question": "Question?",
        "system_prompt": "Policy",
        "context": "{{user_question}}\n{{documents}}\n{{output_json}}",
        "answer_json": {"answer": "string"},
        "documents": [
            {"document_id": "own", "version": "1", "content": "First experiment content.", "enabled": True},
            {"document_id": "private", "version": "1", "content": "DISABLED PRIVATE TEXT", "enabled": False},
            {"document_id": "other", "version": "1", "content": "Last enabled content.", "enabled": True},
        ],
    }


def save(client, data):
    result = client.post("/api/experiment-definitions", json=data)
    assert result.status_code == 201, result.text
    return result.json()


def preview(client, data):
    return client.post(
        "/api/context-preview",
        json={
            "question": data["question"],
            "documents": data["documents"],
            "context_mode": "explicit",
            "context_ids": [],
            "prompt_instructions": data["context"],
            "system_prompt": data["system_prompt"],
            "answer_json": data["answer_json"],
        },
    )


def test_preview_is_read_only_omits_disabled_and_preserves_order(client, db):
    data = definition()
    before = db.one("SELECT count(*) n FROM document_versions")["n"]
    response = preview(client, data)
    assert response.status_code == 200, response.text
    text = response.json()["text"]
    assert "DISABLED PRIVATE TEXT" not in text
    assert text.index("First experiment") < text.index("Last enabled")
    assert [doc["document_id"] for doc in response.json()["documents"]] == ["own", "other"]
    assert db.one("SELECT count(*) n FROM document_versions")["n"] == before
    data["documents"].reverse()
    reordered = preview(client, data).json()["text"]
    assert reordered.index("Last enabled") < reordered.index("First experiment")


def test_experiments_have_independent_documents_with_same_labels(client, monkeypatch):
    first = save(client, definition())
    second_data = definition()
    second_data["title"] = "Other experiment"
    second_data["documents"][0]["content"] = "Independent second experiment."
    second = save(client, second_data)
    assert first["definition"]["documents"][0]["content"] != second["definition"]["documents"][0]["content"]
    edited = deepcopy(first["definition"])
    edited["documents"][0]["content"] = "Edited only here."
    client.put(f"/api/experiment-definitions/{first['id']}", json=edited)
    records = client.get("/api/experiment-definitions").json()
    assert next(item for item in records if item["id"] == second["id"])["definition"] == second["definition"]


def test_owned_run_matches_preview_and_keeps_historical_bytes(client, monkeypatch):
    data = definition()
    saved = save(client, data)
    prepared = preview(client, data).json()
    requests = []

    def generate(settings, body, chunks):
        requests.append(body)
        assert [chunk["document_id"] for chunk in chunks] == ["own", "other"]
        assert "DISABLED PRIVATE TEXT" not in json.dumps(body)
        return {"choices": [{"message": {"content": '{"answer":"response"}'}, "finish_reason": "stop"}]}

    monkeypatch.setattr("traceability_lab.pipeline.generate", generate)
    response = client.post(
        "/api/experiment-definitions/run",
        json={
            "definition": saved["definition"],
            "saved_id": saved["id"],
            "execution": {
                "llm_profile_id": "test-generator",
                "system_prompt": prepared["system_prompt"],
                "llm_context": prepared["text"],
                "context_chunk_ids": prepared["chunk_ids"],
                "repetitions": 2,
            },
        },
    )
    assert response.status_code == 201, response.text
    ids = response.json()["run_ids"]
    run = client.get(f"/api/runs/{ids[0]}").json()
    assert run["status"] == "completed", run.get("error")
    assert run["messages"][-1]["content"] == prepared["text"]
    assert run["snapshot"]["context_edited"] is False
    assert [source["chunk_id"] for source in run["context"]] == prepared["chunk_ids"]
    assert len(requests) == 2
    edited = deepcopy(saved["definition"])
    edited["documents"][0]["content"] = "New version of owned text."
    client.put(f"/api/experiment-definitions/{saved['id']}", json=edited)
    assert client.get(f"/api/runs/{ids[0]}").json()["context"] == run["context"]
    assert not any(doc["document_id"] in {"own", "other"} for doc in client.get("/api/documents").json())


@pytest.mark.parametrize("empty", [False, True])
def test_zero_enabled_documents_can_be_saved_previewed_and_run(client, empty):
    data = definition()
    for doc in data["documents"]:
        doc["enabled"] = False
    if empty:
        data["documents"] = []
    save(client, data)
    response = preview(client, data)
    assert response.status_code == 200, response.text
    assert response.json()["documents"] == []
    assert response.json()["chunk_ids"] == []
    assert "DISABLED PRIVATE TEXT" not in response.json()["text"]
    response = client.post("/api/experiment-definitions/run", json={"definition": data})
    assert response.status_code == 201, response.text
    batch = client.get(f"/api/batches/{response.json()['batch_id']}").json()
    run = client.get(f"/api/runs/{batch['runs'][0]['id']}").json()
    assert run["context"] == []
    assert "First experiment content" not in json.dumps(run["messages"])


def test_duplicate_labels_rejected_within_experiment(client):
    data = definition()
    data["documents"].append(deepcopy(data["documents"][0]))
    assert client.post("/api/experiment-definitions", json=data).status_code == 422


def test_export_import_contains_owned_text_order_and_disabled_state(client):
    saved = save(client, definition())
    exported = client.get(f"/api/experiment-definitions/{saved['id']}/export").json()
    assert exported["format"] == "traceability-experiment-v2"
    assert "document_ids" not in exported["experiment"]
    assert "version_ids" not in exported["experiment"]
    assert [doc["id"] for doc in exported["experiment"]["documents"]] == ["own@1", "private@1", "other@1"]
    imported = client.post("/api/experiment-definitions/import", json=exported)
    assert imported.status_code == 200, imported.text
    assert imported.json()["definition"] == saved["definition"]
    assert len(client.get("/api/experiment-definitions").json()) == 1
    changed_ids = deepcopy(exported)
    changed_ids["source_ids"]["own@1"] = ["old-source-id"]
    warned = client.post("/api/experiment-definitions/import", json=changed_ids)
    assert warned.json()["warnings"]
    assert warned.json()["definition"] == saved["definition"]
    exported["experiment"]["documents"][0]["content"] = "Tampered bytes"
    assert client.post("/api/experiment-definitions/import", json=exported).status_code == 422


def test_legacy_document_copies_remain_available_after_library_edit(client):
    original = next(
        doc
        for doc in client.get("/api/documents").json()
        if doc["document_id"] == "orion" and doc["version"] == "1"
    )
    saved = save(client, {"title": "Legacy copy", "question": "Question?", "version_ids": [original["id"]]})
    client.delete(f"/api/document-versions/{original['id']}")
    record = next(
        item for item in client.get("/api/experiment-definitions").json() if item["id"] == saved["id"]
    )
    assert record["document_copies"][0]["content"] == original["content"]
    assert record["document_copies"][0]["enabled"] is True
    exported = client.get(f"/api/experiment-definitions/{saved['id']}/export").json()
    assert exported["experiment"]["documents"][0]["content"] == original["content"]


def test_owned_documents_support_optional_semantic_evaluator(client, monkeypatch):
    data = definition()
    data.update(output_mode="facts_and_answer", semantic_evaluator={"llm_profile_id": "test-reviewer"})

    def generate(settings, body, chunks):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "facts": [{"fact": "A fact", "source_ids": [chunks[0]["id"]]}],
                                "answer": None,
                            }
                        )
                    },
                    "finish_reason": "stop",
                }
            ]
        }

    def evaluate(settings, body, chunks):
        payload = json.loads(body["messages"][1]["content"])
        if "generated_fact" in payload:
            assert [doc["document_id"] for doc in payload["supplied_sources"]] == ["own"]
            value = {
                "fact_index": 0,
                "expected_fact_indices": [],
                "meaning_match": "not_evaluated",
                "source_support": "supported",
                "explanation": "Passage supports this statement.",
            }
        else:
            assert "output_schema" not in payload
            assert payload["generated_answer"] is None
            value = {
                "answer": {
                    "meaning_match": "not_evaluated",
                    "evidence_support": "insufficient_evidence",
                    "explanation": "No answer.",
                },
                "unsupported_answer": False,
            }
        return {"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]}

    monkeypatch.setattr("traceability_lab.pipeline.generate", generate)
    monkeypatch.setattr("traceability_lab.semantic_evaluation.generate", evaluate)
    response = client.post(
        "/api/experiment-definitions/run",
        json={"definition": data, "execution": {"repetitions": 1, "llm_profile_id": "test-generator"}},
    )
    assert response.status_code == 201, response.text
    evaluation = client.get(f"/api/runs/{response.json()['run_ids'][0]}/evaluation").json()
    assert evaluation["automatic"]["semantic_evaluation"]["status"] == "completed"
