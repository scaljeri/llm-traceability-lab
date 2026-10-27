"""Prompt selection is independent of data; runs retain immutable templates."""

import json
from copy import deepcopy

import pytest

from traceability_lab.prompts import seed_prompts


@pytest.mark.parametrize("operation", ["preview", "run"])
def test_invalid_stored_prompt_has_readable_named_error(client, db, operation):
    prompt = next(item for item in client.get("/api/prompts").json() if item["type"] == "answer")
    prompt["user_prompt"] = "{{user_question}}"
    payload = json.dumps(prompt)
    with db.connect() as connection:
        connection.execute("UPDATE prompts SET payload=? WHERE id=?", (payload, prompt["id"]))
    if operation == "preview":
        response = client.post(
            "/api/context-preview",
            json={
                "question": "Question?",
                "documents": [],
                "context_mode": "explicit",
                "output_mode": "facts_and_answer",
                "answer_prompt_id": prompt["id"],
            },
        )
    else:
        response = client.post(
            "/api/experiment-definitions/run",
            json={
                "definition": {
                    "title": "Test",
                    "question": "Question?",
                    "documents": [],
                    "output_mode": "facts_and_answer",
                },
                "execution": {
                    "repetitions": 1,
                    "llm_profile_id": "test-generator",
                    "answer_prompt_id": prompt["id"],
                },
            },
        )
    assert response.status_code == 422, response.text
    message = response.json()["detail"]
    assert prompt["name"] in message and prompt["id"] in message
    assert "Generation prompts must include {{documents}} exactly once." in message
    assert "Manage prompts" in message
    assert all(text not in message for text in ("input_value", "pydantic.dev", "Value error,"))
    assert db.one("SELECT payload FROM prompts WHERE id=?", (prompt["id"],))["payload"] == payload


def test_prompt_crud_validation_and_seeding(client, db, settings):
    initial = client.get("/api/prompts").json()
    assert {item["type"] for item in initial} == {"answer", "review"}
    review = next(item for item in initial if item["type"] == "review")
    assert "fact_prompt" in review and "answer_prompt" in review
    assert "user_prompt" not in review
    value = {
        "id": "custom-prompt",
        "type": "answer",
        "name": "Custom",
        "system_prompt": "Custom policy",
        "user_prompt": "{{user_question}}\n{{documents}}\n{{output_json}}",
    }
    saved = client.post("/api/prompts", json=value)
    assert saved.status_code == 201, saved.text
    assert client.post("/api/prompts", json=value).status_code == 409
    for invalid in [
        {**value, "user_prompt": "Question: {{user_question}}\nReturn: {{output_json}}"},
        {**value, "user_prompt": "{{documents}}\n{{documents}}"},
        {**value, "user_prompt": "{{expected_answer}}"},
        {**value, "name": " "},
        {**value, "system_prompt": "{{documents}}"},
        {**review, "system_prompt": "{{review_input}}"},
        {**value, "system_prompt": "{{output_json}} {{output_json}}"},
        {**review, "fact_prompt": "No payload"},
        {**review, "fact_prompt": "{{review_input}} {{review_input}}"},
    ]:
        assert client.post("/api/prompts", json=invalid).status_code == 422
    assert client.put("/api/prompts/custom-prompt", json={**value, "name": "Renamed"}).status_code == 200
    assert client.put("/api/prompts/custom-prompt", json={**value, "id": "other"}).status_code == 422
    assert client.delete("/api/prompts/custom-prompt").status_code == 200
    assert client.delete("/api/prompts/custom-prompt").status_code == 404
    assert client.delete("/api/prompts/" + review["id"]).status_code == 200
    seed_prompts(db, settings.prompts_directory)
    assert review["id"] not in {item["id"] for item in client.get("/api/prompts").json()}


@pytest.mark.parametrize("reference_available", [True, False])
def test_selected_prompts_drive_preview_and_both_reviews_and_are_frozen(
    client, db, monkeypatch, reference_available
):
    answer_prompt = {
        "id": "test-answer-prompt",
        "type": "answer",
        "name": "Answer instructions",
        "system_prompt": "Generator only policy\n{{output_json}}",
        "user_prompt": "ASK {{user_question}}\n{{documents}}",
    }
    review_prompt = {
        "id": "test-review-prompt",
        "type": "review",
        "name": "Review instructions",
        "system_prompt": "Facts reviewer policy\n{{output_json}}",
        "fact_prompt": "FACT REVIEW\n{{review_input}}",
        "answer_prompt": "UNSELECTED ANSWER\n{{review_input}}",
    }
    answer_review = {
        **review_prompt,
        "id": "test-answer-review",
        "name": "Answer review instructions",
        "system_prompt": "Answer reviewer policy\n{{output_json}}",
        "fact_prompt": "UNSELECTED FACT\n{{review_input}}",
        "answer_prompt": "ANSWER REVIEW\n{{review_input}}",
    }
    for value in [answer_prompt, review_prompt, answer_review]:
        assert client.post("/api/prompts", json=value).status_code == 201
    fact = {
        "fact": "Ada is the manager.",
        "sources": [{"document_id": "project", "version": "1", "quote": "Manager: Ada."}],
    }
    generated = {"facts": [fact], "answer": "Ada is the manager."}
    definition = {
        "title": "Independent prompts",
        "question": "Who is the manager?",
        "output_mode": "facts_and_answer",
        "system_prompt": "Old experiment policy",
        "context": "Old experiment template",
        "documents": [{"document_id": "project", "version": "1", "content": "Manager: Ada."}],
        "semantic_reference": {"facts": [fact], "answer": "Ada is the manager."}
        if reference_available
        else None,
        "semantic_evaluator": {"llm_profile_id": "test-reviewer"},
    }
    original = deepcopy(definition)
    preview = client.post(
        "/api/context-preview",
        json={
            "question": definition["question"],
            "documents": definition["documents"],
            "context_mode": "explicit",
            "output_mode": "facts_and_answer",
            "answer_prompt_id": answer_prompt["id"],
        },
    )
    assert preview.status_code == 200, preview.text
    preview = preview.json()
    assert preview["system_prompt"].startswith("Generator only policy")
    assert "OUTPUT CONTRACT" not in preview["system_prompt"]
    assert '"quote"' in preview["system_prompt"]
    assert '"quote"' not in preview["text"]
    assert "ASK Who is the manager?" in preview["text"]
    assert preview["text"].count("Manager: Ada.") == 1
    assert preview["text"].count(definition["question"]) == 1
    assert "Old experiment" not in preview["text"]
    assert "{{output_json}}" not in preview["text"]
    calls = []

    def envelope(value):
        return {"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]}

    def generator(settings, body, chunks, **kwargs):
        assert body["messages"][0]["content"] == preview["system_prompt"]
        assert body["messages"][1]["content"] == preview["text"]
        return envelope(generated)

    def reviewer(settings, body, chunks):
        calls.append(body)
        text = body["messages"][1]["content"]
        payload = json.loads(text.split("\n", 1)[1].split("\n\nFormat:", 1)[0])
        policy = body["messages"][0]["content"]
        if not reference_available:
            assert 'meaning_match to "not_evaluated"' in policy
            assert "correctness_score and confidence to null" in policy
        if text.startswith("FACT REVIEW"):
            assert policy.startswith(review_prompt["system_prompt"].split("{{output_json}}")[0])
            assert payload["generated_fact"] == fact
            assert "output_schema" not in payload
            assert '"fact_index"' in policy
            assert "{{output_json}}" not in policy
            return envelope(
                {
                    "fact_index": payload["fact_index"],
                    "expected_fact_indices": [0] if reference_available else [],
                    "meaning_match": "equivalent" if reference_available else "not_evaluated",
                    "source_support": "supported",
                    "correctness_score": 100 if reference_available else None,
                    "confidence": 95 if reference_available else None,
                    "explanation": "Matches.",
                }
            )
        assert text.startswith("ANSWER REVIEW")
        assert policy.startswith(answer_review["system_prompt"].split("{{output_json}}")[0])
        assert "{{output_json}}" not in policy
        assert policy.count("Format:") == 1
        assert payload["generated_answer"] == generated["answer"]
        assert set(payload) == {"question", "generated_answer", "expected_answer", "documents"}
        return envelope(
            {
                "answer": {
                    "meaning_match": "equivalent" if reference_available else "not_evaluated",
                    "evidence_support": "supported",
                    "correctness_score": 100 if reference_available else None,
                    "confidence": 95 if reference_available else None,
                    "explanation": "Matches.",
                },
                "unsupported_answer": False,
            }
        )

    monkeypatch.setattr("traceability_lab.pipeline.generate", generator)
    monkeypatch.setattr("traceability_lab.semantic_evaluation.generate", reviewer)
    response = client.post(
        "/api/experiment-definitions/run",
        json={
            "definition": definition,
            "execution": {
                "repetitions": 1,
                "llm_profile_id": "test-generator",
                "answer_prompt_id": answer_prompt["id"],
                "review_facts_prompt_id": review_prompt["id"],
                "review_answer_prompt_id": answer_review["id"],
                "llm_context": "STALE CLIENT PREVIEW",
                "context_chunk_ids": [],
            },
        },
    )
    assert response.status_code == 201, response.text
    batch = response.json()
    assert len(calls) == 2
    assert calls[0]["messages"][0]["content"].startswith(
        review_prompt["system_prompt"].split("{{output_json}}")[0]
    )
    assert calls[1]["messages"][0]["content"].startswith(
        answer_review["system_prompt"].split("{{output_json}}")[0]
    )
    stored = client.get("/api/runs/" + batch["run_ids"][0]).json()
    assert stored["messages"][0]["content"] == preview["system_prompt"]
    assert stored["messages"][1]["content"] == preview["text"]
    snapshot = deepcopy(stored["snapshot"])
    assert snapshot["experiment"]["prompts"]["answer"]["name"] == answer_prompt["name"]
    assert snapshot["experiment"]["prompts"]["review_facts"]["fact_prompt"] == review_prompt["fact_prompt"]
    assert (
        snapshot["experiment"]["prompts"]["review_answer"]["answer_prompt"] == answer_review["answer_prompt"]
    )
    assessment = client.get("/api/runs/" + stored["id"] + "/evaluation").json()
    assert assessment["automatic"]["semantic_evaluation"]["prompt_version"] == "prompt-library-v2"
    assert assessment["automatic"]["semantic_evaluation"]["status"] == "completed", assessment["automatic"][
        "semantic_evaluation"
    ].get("error")
    assert assessment["automatic"]["semantic_evaluation"]["calls"][0]["request_body"] == calls[0]
    provenance = client.get(f"/api/runs/{stored['id']}/provenance").json()
    from traceability_lab.documents import digest

    payload = db.one("SELECT payload FROM run_evaluations WHERE run_id=?", (stored["id"],))["payload"]
    evaluation_entity = provenance["entity"][f"lab:evaluation-{stored['id']}"]
    assert "prov:value" not in evaluation_entity
    assert evaluation_entity["lab:sha256"] == digest(payload)
    aggregate = provenance["activity"][f"lab:evaluation-activity-{stored['id']}"]
    assert aggregate["prov:startTime"] == assessment["automatic"]["evaluation_started_at"]
    assert aggregate["prov:endTime"] == assessment["automatic"]["evaluation_finished_at"]
    for agent in provenance["agent"].values():
        assert agent["prov:type"]["$"] == "prov:SoftwareAgent"
        assert "prov:value" not in agent
    assert not set(provenance["entity"]) & set(provenance["activity"])
    prompt_entity = provenance["entity"][f"lab:prompt-{stored['id']}"]
    for key in ("id", "name", "sha256"):
        assert prompt_entity[f"lab:prompt_{key}"] == snapshot["experiment"]["prompts"]["answer"][key]
    assert f"lab:reviewer-{stored['id']}" in provenance["agent"]
    provn = client.get(f"/api/runs/{stored['id']}/provenance?format=provn")
    assert provn.status_code == 200
    assert f"lab:reviewer-{stored['id']}" in provn.text
    for index, phase in enumerate(("fact", "answer")):
        identity = f"{stored['id']}-{index}"
        activity = provenance["activity"][f"lab:review-{identity}"]
        assert activity["lab:phase"] == phase
        assert (
            activity["prov:startTime"]
            == assessment["automatic"]["semantic_evaluation"]["calls"][index]["started_at"]
        )
        request = provenance["entity"][f"lab:review-request-{identity}"]
        assert json.loads(request["prov:value"]) == calls[index]
        result = provenance["entity"][f"lab:review-result-{identity}"]
        recorded = assessment["automatic"]["semantic_evaluation"]["calls"][index]
        assert json.loads(result["prov:value"]) == recorded["response"]
        assert json.loads(result["lab:judgment"]) == recorded["parsed_answer"]
        assert (
            sum(entity.get("prov:value") == request["prov:value"] for entity in provenance["entity"].values())
            == 1
        )
        assert request["lab:prompt_id"] == (review_prompt if index == 0 else answer_review)["id"]
        assert any(
            item["prov:activity"] == f"lab:review-{identity}"
            and item["prov:entity"] == f"lab:review-result-{identity}"
            for item in provenance["wasGeneratedBy"].values()
        )
    assert definition == original
    assert db.one("SELECT payload FROM run_evaluations WHERE run_id=?", (stored["id"],))["payload"] == payload
    # Historical records have no measured assembly times; semantic call times
    # must not be substituted for those missing measurements.
    from traceability_lab.provenance import export_provenance

    original_rows = db.rows

    def historical_rows(query, parameters=()):
        rows = original_rows(query, parameters)
        if query == "SELECT * FROM run_evaluations WHERE run_id=?":
            legacy = json.loads(payload)
            legacy.pop("evaluation_started_at")
            legacy.pop("evaluation_finished_at")
            return [{**dict(rows[0]), "payload": json.dumps(legacy)}]
        return rows

    with monkeypatch.context() as patch:
        patch.setattr(db, "rows", historical_rows)
        historical = export_provenance(db, stored["id"])
    historical_activity = historical["activity"][f"lab:evaluation-activity-{stored['id']}"]
    assert "prov:startTime" not in historical_activity
    assert "prov:endTime" not in historical_activity
    from traceability_lab.reproducibility import reproducibility_record

    baseline_hash = reproducibility_record(stored, assessment["automatic"])["configuration_sha256"]
    changed_review = deepcopy(assessment["automatic"])
    changed_review["semantic_evaluation"]["user_prompts"]["fact"] += " Different instructions."
    assert reproducibility_record(stored, changed_review)["configuration_sha256"] != baseline_hash
    for value in [answer_prompt, review_prompt]:
        assert client.put("/api/prompts/" + value["id"], json={**value, "name": "Changed"}).status_code == 200
        assert client.delete("/api/prompts/" + value["id"]).status_code == 200
    assert client.get("/api/runs/" + stored["id"]).json()["snapshot"] == snapshot
    assert (
        client.post(
            "/api/context-preview",
            json={
                "question": "Who?",
                "context_mode": "explicit",
                "documents": [],
                "answer_prompt_id": review_prompt["id"],
            },
        ).status_code
        == 404
    )


def test_three_source_files_compose_one_review_set():
    from pathlib import Path

    from traceability_lab.config import PROJECT_ROOT
    from traceability_lab.prompts import read_prompt_files

    directory = PROJECT_ROOT / "prompts"
    assert {path.name for path in directory.iterdir()} == {
        "generation_prompt.json",
        "fact_review_prompt.json",
        "answer_review_prompt.json",
    }
    values = read_prompt_files(directory)
    assert len(values) == 2
    review = next(value for value in values if value["type"] == "review")
    facts = json.loads((directory / "fact_review_prompt.json").read_text())
    answer = json.loads((directory / "answer_review_prompt.json").read_text())
    assert review["system_prompt"] == facts["system_prompt"] == answer["system_prompt"]
    assert review["fact_prompt"] == facts["user_prompt"]
    assert review["answer_prompt"] == answer["user_prompt"]
    assert review["fact_prompt"] != review["answer_prompt"]
    for prompt in values:
        assert prompt["system_prompt"].count("{{output_json}}") == 1
        for field in ("user_prompt", "fact_prompt", "answer_prompt"):
            assert "{{output_json}}" not in prompt.get(field, "")
    assert Path("prompts/extracted").exists() is False


def test_incomplete_or_conflicting_review_source_files_are_not_seeded(tmp_path):
    from traceability_lab.db import Database
    from traceability_lab.prompts import seed_prompts

    db = Database(tmp_path / "lab.sqlite3")
    db.initialize()
    directory = tmp_path / "prompts"
    directory.mkdir()
    facts = {
        "format": "traceability-review-stage-v1",
        "id": "review",
        "name": "Review",
        "stage": "facts",
        "system_prompt": "Shared",
        "user_prompt": "Facts {{review_input}}",
    }
    answer = {**facts, "stage": "answer", "user_prompt": "Answer {{review_input}}"}
    fact_file = directory / "facts.json"
    fact_file.write_text(json.dumps(facts))
    with pytest.raises(ValueError, match="both facts and answer"):
        seed_prompts(db, directory)
    assert not db.rows("SELECT * FROM prompts")
    assert not db.rows("SELECT * FROM prompt_library")
    answer_file = directory / "generation_prompt.json"
    answer_file.write_text(json.dumps({**answer, "system_prompt": "Different"}))
    with pytest.raises(ValueError, match="share its name and system prompt"):
        seed_prompts(db, directory)
    assert not db.rows("SELECT * FROM prompts")
    answer_file.write_text(json.dumps(answer))
    duplicate = directory / "duplicate.json"
    duplicate.write_text(json.dumps(facts))
    with pytest.raises(ValueError, match="Duplicate facts"):
        seed_prompts(db, directory)
    duplicate.unlink()
    seed_prompts(db, directory)
    assert len(db.rows("SELECT * FROM prompts")) == 1


def test_new_wildcards_do_not_expand_inserted_document_text():
    from traceability_lab.llm import build_messages

    text = "Source literally says {{user_question}} and {{documents}}."
    messages = build_messages(
        "Who?",
        [{"document_id": "doc", "version": "1", "title": "Document", "text": text}],
        output_mode="facts_and_answer",
        prompt_instructions="{{user_question}}\n{{documents}}\n{{output_json}}",
        answer_json={"answer": "string"},
    )
    assert text in messages[-1]["content"]
    assert messages[-1]["content"].count("Who?") == 1
    assert messages[-1]["content"].count("## BEGIN SOURCE 1") == 1


@pytest.mark.parametrize("old_name", ["question", "sources", "json", "..."])
def test_old_placeholder_names_are_rejected(old_name):
    from traceability_lab.llm import build_messages
    from traceability_lab.prompts import PromptInput

    template = "{{" + old_name + "}}"
    with pytest.raises(ValueError, match="Unknown prompt placeholder"):
        PromptInput(type="answer", name="Old template", user_prompt=template)
    with pytest.raises(ValueError, match="Unknown prompt placeholder"):
        build_messages("Question?", [], prompt_instructions=template)


def test_example_experiments_contain_data_not_prompt_templates():
    from traceability_lab.config import PROJECT_ROOT
    from traceability_lab.initial_experiments import read_initial_experiments

    paths = list((PROJECT_ROOT / "experiments").glob("*.json"))
    assert paths
    for path in paths:
        value = json.loads(path.read_text())
        assert value["format"] == "traceability-experiment-v2"
        data = value.get("experiment", value)
        assert "version_ids" not in data
        assert not {"system_prompt", "context", "answer_json"} & data.keys(), path.name
        assert data["question"] and isinstance(data["documents"], list)
        assert "answer" in data["semantic_reference"]
        assert "conclusion" not in data["semantic_reference"]
    assert len(read_initial_experiments(PROJECT_ROOT / "experiments")) == len(paths)


def test_empty_prompt_directory_can_be_seeded_when_files_are_added(tmp_path):
    from traceability_lab.db import Database
    from traceability_lab.prompts import list_prompts

    db = Database(tmp_path / "lab.sqlite3")
    db.initialize()
    directory = tmp_path / "prompts"
    directory.mkdir()
    seed_prompts(db, directory)
    assert db.rows("SELECT * FROM prompt_library") == []
    value = {"id": "added", "type": "answer", "name": "Added later", "user_prompt": "{{documents}}"}
    (directory / "generation_prompt.json").write_text(json.dumps(value))
    seed_prompts(db, directory)
    assert list_prompts(db)[0]["id"] == "added"
    assert db.rows("SELECT * FROM prompt_library")


def test_fact_library_selection_keeps_builtin_answer_output_instructions(client, monkeypatch):
    from traceability_lab.semantic_evaluation import ANSWER_OUTPUT_FORMAT

    review = next(prompt for prompt in client.get("/api/prompts").json() if prompt["type"] == "review")

    def generator(*args, **kwargs):
        return {
            "choices": [{"message": {"content": '{"facts": [], "answer": null}'}, "finish_reason": "stop"}]
        }

    calls = []

    def reviewer(settings, body, chunks):
        calls.append(body)
        assert ANSWER_OUTPUT_FORMAT in body["messages"][0]["content"]
        value = {
            "answer": {
                "meaning_match": "not_equivalent",
                "evidence_support": "insufficient_evidence",
                "correctness_score": 0,
                "confidence": 100,
                "explanation": "Expected answer is missing.",
            },
            "unsupported_answer": False,
        }
        return {"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]}

    monkeypatch.setattr("traceability_lab.pipeline.generate", generator)
    monkeypatch.setattr("traceability_lab.semantic_evaluation.generate", reviewer)
    response = client.post(
        "/api/experiment-definitions/run",
        json={
            "definition": {
                "title": "Only facts template selected",
                "question": "Question?",
                "documents": [],
                "output_mode": "facts_and_answer",
                "semantic_reference": {"facts": [], "answer": "Expected"},
                "semantic_evaluator": {"llm_profile_id": "test-reviewer"},
            },
            "execution": {
                "repetitions": 1,
                "llm_profile_id": "test-generator",
                "review_facts_prompt_id": review["id"],
            },
        },
    )
    assert response.status_code == 201, response.text
    assert len(calls) == 1
    run_id = response.json()["run_ids"][0]
    assessment = client.get(f"/api/runs/{run_id}/evaluation").json()
    assert assessment["automatic"]["semantic_evaluation"]["status"] == "completed", assessment["automatic"][
        "semantic_evaluation"
    ].get("error")
