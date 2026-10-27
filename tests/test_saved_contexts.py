"""Saved operator contexts are independent from source evidence and frozen per run."""

import pytest
from fastapi.testclient import TestClient
from legacy_api import legacy_app as create_app


def add(client, title, content, active=True):
    response = client.post("/api/contexts", json={"title": title, "content": content, "active": active})
    assert response.status_code == 201
    return response.json()


@pytest.mark.parametrize(
    ("output_mode", "instructions"),
    [
        ("custom", "Question: {{user_question}}\nDocuments: {{documents}}"),
        ("facts_and_answer", "Question: {{user_question}}\nDocuments: {{documents}}"),
        ("facts_and_answer", "Summarize the evidence."),
    ],
)
def test_unified_context_reaches_direct_and_prepared_requests_once(client, output_mode, instructions):
    first = add(client, "First", "FIRST_OPERATOR_MARKER")
    second = add(client, "Second", "SECOND_OPERATOR_MARKER")
    add(client, "Inactive", "INACTIVE_OPERATOR_MARKER", False)
    version = client.get("/api/documents").json()[0]["id"]
    question = {
        "question": "What is established?",
        "version_ids": [version],
        "context_mode": "explicit",
        "output_mode": output_mode,
        "prompt_instructions": instructions,
        "context_ids": [second["id"], first["id"]],
    }
    preview_response = client.post("/api/context-preview", json=question)
    assert preview_response.status_code == 200, preview_response.text
    preview = preview_response.json()
    assert preview["text"].index("SECOND_OPERATOR_MARKER") < preview["text"].index("FIRST_OPERATOR_MARKER")
    for prepared in (False, True):
        payload = dict(question)
        if prepared:
            payload.update(llm_context=preview["text"], context_chunk_ids=preview["chunk_ids"])
        response = client.post("/api/questions", json=payload)
        assert response.status_code == 201, response.text
        run = response.json()
        assert run["status"] == "completed", run["error"]
        assert run["messages"][-1]["content"] == preview["text"]
        assert run["snapshot"]["context_edited"] is False
        assert run["snapshot"]["request_body"]["messages"] == run["messages"]
        for marker in ("FIRST_OPERATOR_MARKER", "SECOND_OPERATOR_MARKER"):
            assert preview["text"].count(marker) == 1
        assert "INACTIVE_OPERATOR_MARKER" not in str(run["messages"])


def test_active_context_order_editing_and_restart(client, settings):
    first = add(client, "First", "First instruction")
    second = add(client, "Second", "Second instruction")
    add(client, "Inactive", "Do not include", False)
    contexts = client.get("/api/contexts").json()
    order = [second["id"], first["id"], contexts[2]["id"]]
    assert client.post("/api/contexts/order", json={"ids": order}).status_code == 200
    run = client.post("/api/questions", json={"question": "Test?"}).json()
    assert run["status"] == "completed", run["error"]
    assert [item["id"] for item in run["snapshot"]["supplemental_contexts"]] == order[:2]
    assert run["messages"][1]["content"].index("Second instruction") < run["messages"][1]["content"].index(
        "First instruction"
    )
    assert "Do not include" not in str(run["messages"])
    assert run["snapshot"]["prompt_version"].endswith("+contexts-v1")
    assert (
        client.put(
            f"/api/contexts/{first['id']}", json={"title": "Changed", "content": "Edited", "active": False}
        ).status_code
        == 200
    )
    assert client.get(f"/api/runs/{run['id']}").json()["snapshot"] == run["snapshot"]
    provenance = client.get(f"/api/runs/{run['id']}/provenance").json()
    assert "First instruction" in str(provenance)
    with TestClient(create_app(settings)) as restarted:
        assert [item["id"] for item in restarted.get("/api/contexts").json()] == order
    empty = client.post("/api/questions", json={"question": "Test?", "context_ids": []}).json()
    assert empty["snapshot"]["supplemental_contexts"] == []


def test_context_validation_and_explicit_order(client):
    first = add(client, "One", "One")
    second = add(client, "Two", "Two", False)
    assert client.post("/api/contexts", json={"title": " ", "content": "x"}).status_code == 422
    assert client.post("/api/contexts/order", json={"ids": [first["id"], first["id"]]}).status_code == 422
    assert client.post("/api/contexts/order", json={"ids": [first["id"]]}).status_code == 422
    assert (
        client.post("/api/questions", json={"question": "Test?", "context_ids": ["missing"]}).status_code
        == 422
    )
    run = client.post(
        "/api/questions", json={"question": "Test?", "context_ids": [second["id"], first["id"]]}
    ).json()
    assert [item["title"] for item in run["snapshot"]["supplemental_contexts"]] == ["Two", "One"]


def test_batch_freezes_contexts_before_first_generation(client, monkeypatch):
    first = add(client, "Frozen", "Original context")
    from traceability_lab import pipeline

    generate = pipeline.generate
    count = 0

    def edit_during_generation(*args):
        nonlocal count
        count += 1
        if count == 1:
            client.put(
                f"/api/contexts/{first['id']}",
                json={"title": "Changed", "content": "Changed context", "active": True},
            )
        return generate(*args)

    monkeypatch.setattr(pipeline, "generate", edit_during_generation)
    batch = client.post("/test-scenarios/complete/run", json={"repetitions": 2}).json()
    for identity in batch["run_ids"]:
        run = client.get(f"/api/runs/{identity}").json()
        assert run["snapshot"]["supplemental_contexts"][0]["content"] == "Original context"


def test_run_context_override_is_exact_and_never_changes_saved_contexts(client):
    original = add(client, "Original", "Original saved instruction")
    before = client.get("/api/contexts").json()
    draft = "  Temporary context 👩🏽‍🔬\nOnly this run.  "
    run = client.post("/api/questions", json={"question": "Test?", "context_override": draft}).json()
    assert run["status"] == "completed", run["error"]
    assert run["snapshot"]["question_configuration"]["context_override"] == draft
    assert run["snapshot"]["supplemental_contexts"][0]["content"] == draft
    assert run["snapshot"]["supplemental_contexts"][0]["run_only"] is True
    assert run["messages"][1]["content"].endswith(draft)
    assert client.get("/api/contexts").json() == before
    next_run = client.post("/api/questions", json={"question": "Original again?"}).json()
    assert next_run["snapshot"]["supplemental_contexts"][0]["id"] == original["id"]
    assert next_run["snapshot"]["supplemental_contexts"][0]["content"] == "Original saved instruction"


def test_empty_run_context_override_disables_context_without_changing_activation(client):
    add(client, "Active", "Still active")
    run = client.post("/api/questions", json={"question": "Test?", "context_override": ""}).json()
    assert run["snapshot"]["supplemental_contexts"] == []
    assert client.get("/api/contexts").json()[0]["active"] == 1


def test_experiment_batch_freezes_temporary_context_without_saving_it(client):
    add(client, "Original", "Do not change")
    before = client.get("/api/contexts").json()
    response = client.post(
        "/test-scenarios/version/run", json={"repetitions": 2, "context_override": "Temporary variant"}
    )
    assert response.status_code == 201, response.text
    runs = [client.get("/api/runs/" + identity).json() for identity in response.json()["run_ids"]]
    assert runs[0]["snapshot"]["supplemental_contexts"] == runs[1]["snapshot"]["supplemental_contexts"]
    for run in runs:
        assert run["snapshot"]["experiment"]["overrides"]["context_override"] == "Temporary variant"
        assert "Temporary variant" in run["messages"][1]["content"]
        criterion = client.get(f"/api/runs/{run['id']}/evaluation").json()["automatic"]["criteria"][0]
        assert criterion["applicability"] == "undetermined"
        assert criterion["result"] == "not_evaluated"
    assert client.get("/api/contexts").json() == before


def test_run_context_override_length_is_validated(client):
    response = client.post("/api/questions", json={"question": "Test?", "context_override": "x" * 100001})
    assert response.status_code == 422
    assert client.get("/api/runs").json() == []
