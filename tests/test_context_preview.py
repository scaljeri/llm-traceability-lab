"""The visible context is frozen before generation, without changing source records."""

from fastapi.testclient import TestClient

from traceability_lab.main import create_app


def selection(client, mode="explicit"):
    docs = client.get("/api/documents").json()
    ids = [
        next(d["id"] for d in docs if d["document_id"] == "orion" and d["version"] == v) for v in ["1", "2"]
    ]
    return {
        "question": "What is the current budget for Project Orion?",
        "context_mode": mode,
        "version_ids": ids,
    }


def test_preview_includes_versions_in_order_and_no_generation(client):
    config = selection(client)
    before = client.get("/api/runs").json()
    preview = client.post("/api/context-preview", json=config)
    assert preview.status_code == 200, preview.text
    text = preview.json()["text"]
    assert text.index("version: 1;") < text.index("version: 2;")
    assert "2,8" in text and "2,4" in text
    assert client.get("/api/runs").json() == before
    config["version_ids"].reverse()
    reversed_text = client.post("/api/context-preview", json=config).json()["text"]
    assert reversed_text.index("version: 2;") < reversed_text.index("version: 1;")


def test_exact_edited_context_and_question_survive_restart(client, settings):
    config = selection(client)
    documents = client.get("/api/documents").json()
    preview = client.post("/api/context-preview", json=config).json()
    edited = "  SOURCES\nResearcher supplied €9M.\n\n Unicode: Ω\n"
    request = {**config, "llm_context": edited, "context_chunk_ids": preview["chunk_ids"]}
    run = client.post("/api/questions", json=request).json()
    assert run["status"] == "completed", run.get("error")
    assert run["messages"][-1]["content"] == edited + "\n\nQUESTION\n" + config["question"]
    assert run["snapshot"]["request_body"]["messages"] == run["messages"]
    assert run["snapshot"]["context_edited"] is True
    assert run["context"] == []  # Edited bytes cannot be credited to immutable source chunks.
    assert client.get("/api/documents").json() == documents
    with TestClient(create_app(settings)) as restarted:
        assert restarted.get(f"/api/runs/{run['id']}").json()["messages"] == run["messages"]


def test_unchanged_context_retains_sources_and_instructions_without_duplication(client):
    client.post("/api/contexts", json={"title": "Instruction", "content": "Unique operator instruction"})
    config = selection(client)
    preview = client.post("/api/context-preview", json=config).json()
    assert "Unique operator instruction" in preview["text"]
    run = client.post(
        "/api/questions",
        json={**config, "llm_context": preview["text"], "context_chunk_ids": preview["chunk_ids"]},
    ).json()
    assert run["snapshot"]["context_edited"] is False
    assert [c["chunk_id"] for c in run["context"]] == preview["chunk_ids"]
    assert str(run["messages"]).count("Unique operator instruction") == 1
    assert run["messages"][-1]["content"].startswith(preview["text"])


def test_retrieval_preview_is_used_without_another_retrieval(client, monkeypatch):
    config = selection(client, "retrieval")
    preview = client.post("/api/context-preview", json=config).json()

    def forbidden(*args, **kwargs):
        raise AssertionError("Run must not retrieve again")

    monkeypatch.setattr("traceability_lab.pipeline.retrieve", forbidden)
    run = client.post(
        "/api/questions",
        json={**config, "llm_context": preview["text"], "context_chunk_ids": preview["chunk_ids"]},
    ).json()
    assert run["status"] == "completed", run.get("error")
    assert [c["chunk_id"] for c in run["context"]] == preview["chunk_ids"]
    assert run["messages"][-1]["content"] == preview["text"] + "\n\nQUESTION\n" + config["question"]


def test_edited_experiment_context_invalidates_original_expectations(client):
    config = selection(client)
    preview = client.post("/api/context-preview", json=config).json()
    result = client.post(
        "/test-scenarios/version/run",
        json={**config, "llm_context": "", "context_chunk_ids": preview["chunk_ids"], "repetitions": 2},
    )
    assert result.status_code == 201, result.text
    for identity in result.json()["run_ids"]:
        run = client.get(f"/api/runs/{identity}").json()
        assert run["snapshot"]["experiment"]["modified"] is True
        assert run["snapshot"]["experiment"]["overrides"]["llm_context"] == ""
        criteria = client.get(f"/api/runs/{identity}/evaluation").json()["automatic"]["criteria"]
        assert criteria and all(c["applicability"] == "undetermined" for c in criteria)
        assert all(c["result"] == "not_evaluated" for c in criteria)
        assert run["messages"][-1]["content"] == "\n\nQUESTION\n" + config["question"]


def test_preview_source_ids_must_match_selected_versions(client):
    config = selection(client)
    response = client.post(
        "/api/questions", json={**config, "llm_context": "fake", "context_chunk_ids": ["missing"]}
    )
    assert response.status_code == 422
    response = client.post("/api/questions", json={**config, "llm_context": "fake"})
    assert response.status_code == 422


def test_unchanged_experiment_preview_preserves_reference_criteria(client):
    config = selection(client)
    case = next(c for c in client.get("/test-scenarios").json() if c["id"] == "version")
    config.update(
        prompt_instructions=case["prompt_instructions"], version_instructions=case["version_instructions"]
    )
    preview = client.post("/api/context-preview", json=config).json()
    result = client.post(
        "/test-scenarios/version/run",
        json={**config, "llm_context": preview["text"], "context_chunk_ids": preview["chunk_ids"]},
    )
    assert result.status_code == 201, result.text
    identity = result.json()["run_ids"][0]
    run = client.get(f"/api/runs/{identity}").json()
    assert run["status"] == "completed"
    assert run["snapshot"]["experiment"]["modified"] is False
    criteria = client.get(f"/api/runs/{identity}/evaluation").json()["automatic"]["criteria"]
    assert all(
        c["applicability"] == ("not_applicable" if c["id"] == "retrieval_evidence" else "applicable")
        for c in criteria
    )


def test_preview_instructions_match_actual_system_role(client):
    config = {
        **selection(client),
        "version_instructions": True,
        "prompt_instructions": "Distinguish both versions.",
    }
    preview = client.post("/api/context-preview", json=config).json()
    instructions = preview["system_prompt"]
    assert "Follow the instructions." in instructions
    assert "higher sequence number" in instructions
    assert "Distinguish both versions." in instructions
    assert config["question"] not in instructions
    run = client.post(
        "/api/questions",
        json={
            **config,
            "llm_context": preview["text"],
            "context_chunk_ids": preview["chunk_ids"],
            "system_prompt": instructions,
        },
    ).json()
    assert run["status"] == "completed", run.get("error")
    assert run["messages"][0] == {"role": "system", "content": instructions}
    assert run["snapshot"]["system_prompt_edited"] is False
    removed_mode = client.post("/api/context-preview", json={**config, "mode": "unrestricted"})
    assert removed_mode.status_code == 422


def test_edited_system_instructions_exact_and_evaluation_requires_review(client, settings):
    config = selection(client)
    preview = client.post("/api/context-preview", json=config).json()
    edited = "  Custom system instructions Ω\nUse the older document.\n"
    result = client.post(
        "/test-scenarios/version/run",
        json={
            **config,
            "system_prompt": edited,
            "llm_context": preview["text"],
            "context_chunk_ids": preview["chunk_ids"],
        },
    )
    assert result.status_code == 201, result.text
    identity = result.json()["run_ids"][0]
    run = client.get(f"/api/runs/{identity}").json()
    assert run["messages"][0] == {"role": "system", "content": edited}
    assert run["snapshot"]["system_prompt_edited"] is True
    experiment = run["snapshot"]["experiment"]
    assert experiment["modified"] is True
    assert experiment["overrides"]["system_prompt"] == edited
    assert experiment["original_defaults"]["system_prompt"] != edited
    evaluation = client.get(f"/api/runs/{identity}/evaluation").json()
    assert all(
        c["applicability"] == "undetermined" and c["result"] == "not_evaluated"
        for c in evaluation["automatic"]["criteria"]
    )
    with TestClient(create_app(settings)) as restarted:
        assert restarted.get(f"/api/runs/{identity}").json()["messages"][0]["content"] == edited
    assert (
        client.post("/api/context-preview", json=config).json()["system_prompt"] == preview["system_prompt"]
    )


def test_empty_system_prompt_is_an_exact_override(client):
    run = client.post("/api/questions", json={"question": "Budget?", "system_prompt": ""}).json()
    assert run["messages"][0] == {"role": "system", "content": ""}
    assert run["snapshot"]["system_prompt_edited"] is True
    assert (
        client.post("/api/questions", json={"question": "Budget?", "system_prompt": "x" * 20_001}).status_code
        == 422
    )
