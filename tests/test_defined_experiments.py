"""Eigen experimenten: exacte invoer, gescheiden verwachtingen en vaste resultaten."""

import json

import pytest
from fastapi.testclient import TestClient

from traceability_lab.main import create_app


def definition(client):
    docs = client.get("/api/documents").json()
    ids = {doc["version"]: doc["id"] for doc in docs if doc["document_id"] == "orion"}
    return {
        "title": "My budget experiment",
        "documents": None,
        "output_mode": "custom",
        "semantic_reference": None,
        "semantic_evaluator": None,
        "system_prompt": "",
        "question": "What is the current budget for Project Orion?",
        "version_ids": [ids["2"], ids["1"]],
        "context": "Return a JSON object containing budget_eur and supporting_source_id.",
        "answer_json": {"budget_eur": "number or null", "supporting_source_id": "source ID or null"},
        "validation_json": {
            "budget_eur": 2800000,
            "supporting_source_id": {"$source": {"document_id": "orion", "version": "2"}},
        },
    }


def model(monkeypatch, *, budget=2800000, version="2", raw=None, finish="stop"):
    def generate(settings, body, chunks):
        assert "2800000" not in json.dumps(body)
        assert "validation_json" not in json.dumps(body)
        assert [chunk["version"] for chunk in chunks] == ["2", "1"]
        source = next(chunk["id"] for chunk in chunks if chunk["version"] == version)
        text = (
            raw
            if raw is not None
            else json.dumps({"budget_eur": budget, "supporting_source_id": f"[{source}]"})
        )
        return {"choices": [{"message": {"content": text}, "finish_reason": finish}]}

    monkeypatch.setattr("traceability_lab.pipeline.generate", generate)
    monkeypatch.setattr("traceability_lab.pipeline.retrieve", lambda *a: pytest.fail("No retrieval allowed"))


def execute(client, data, saved_id=None, **execution):
    response = client.post(
        "/api/experiment-definitions/run",
        json={
            "definition": data,
            "saved_id": saved_id,
            "execution": {"repetitions": 1, "llm_profile_id": "test-generator", **execution},
        },
    )
    assert response.status_code == 201, response.text
    ids = response.json()["run_ids"]
    return [client.get(f"/api/runs/{identity}/report").json() for identity in ids]


def checks(report):
    return {item["id"]: item for item in report["evaluation"]["automatic"]["criteria"]}


def test_saved_definition_revision_restart_and_frozen_run(client, settings, monkeypatch):
    data = definition(client)
    saved = client.post("/api/experiment-definitions", json=data).json()
    assert saved["revision"] == 1
    model(monkeypatch)
    report = execute(client, data, saved_id=saved["id"])[0]
    assert report["run"]["snapshot"]["experiment"]["definition_version"] == 1
    assert all(item["result"] == "pass" for item in checks(report).values())
    assert report["run"]["snapshot"]["experiment"]["definition"] == data
    assert [item["version"] for item in report["run"]["context"]] == ["2", "1"]
    assert report["evaluation"]["retrieval_recall_at_k"] is None
    changed = {**data, "title": "Changed", "validation_json": {"budget_eur": 1}}
    updated = client.put(f"/api/experiment-definitions/{saved['id']}", json=changed).json()
    assert updated["revision"] == 2
    with TestClient(create_app(settings)) as restarted:
        assert restarted.get("/api/experiment-definitions").json()[0]["definition"] == changed
        assert restarted.get(f"/api/runs/{report['run']['id']}/report").json() == report


@pytest.mark.parametrize(
    ("budget", "version", "failed"),
    [
        (2400000, "2", "budget_eur"),
        (2800000, "1", "supporting_source_id"),
        ("2800000", "2", "budget_eur"),
        (True, "2", "budget_eur"),
        (None, "2", "budget_eur"),
    ],
)
def test_independent_value_and_source_checks(client, monkeypatch, budget, version, failed):
    model(monkeypatch, budget=budget, version=version)
    result = checks(execute(client, definition(client))[0])
    assert result[failed]["result"] == "fail"
    other = "budget_eur" if failed == "supporting_source_id" else "supporting_source_id"
    assert result[other]["result"] == "pass"


def test_numeric_notation_and_repetitions(client, monkeypatch):
    model(monkeypatch, budget=2800000.0)
    reports = execute(client, definition(client), repetitions=3)
    assert len({report["run"]["trace_id"] for report in reports}) == 3
    assert len({report["run"]["batch_id"] for report in reports}) == 1
    assert all(checks(report)["budget_eur"]["result"] == "pass" for report in reports)


@pytest.mark.parametrize("raw", ["Budget is 2800000.", '{"budget_eur":1,"budget_eur":2800000}', "[]"])
def test_ambiguous_output_requires_review(client, monkeypatch, raw):
    model(monkeypatch, raw=raw)
    result = checks(execute(client, definition(client))[0])
    assert result["answer_json"]["result"] == "fail"
    assert result["budget_eur"]["result"] == "requires_review"


def test_missing_value_is_distinct_from_expected_null(client, monkeypatch):
    data = definition(client)
    data["validation_json"] = {"budget_eur": None}
    model(monkeypatch, budget=None)
    assert checks(execute(client, data)[0])["budget_eur"]["result"] == "pass"
    model(monkeypatch, raw='{"supporting_source_id":null}')
    assert checks(execute(client, data)[0])["budget_eur"]["result"] == "fail"


def test_invalid_json_constant_is_not_expected_null(client, monkeypatch):
    data = definition(client)
    data["validation_json"] = {"budget_eur": None}
    model(monkeypatch, raw='{"budget_eur":NaN,"supporting_source_id":null}')
    result = checks(execute(client, data)[0])
    assert result["budget_eur"]["result"] == "requires_review"


def test_overflowing_answer_number_does_not_abort_repetitions(client, monkeypatch):
    raw = '{"budget_eur":1e999,"supporting_source_id":null}'
    model(monkeypatch, raw=raw)
    reports = execute(client, definition(client), repetitions=2)
    assert len(reports) == 2
    for report in reports:
        assert report["run"]["answer"] == raw
        assert report["run"]["status"] == "completed"
        assert checks(report)["answer_json"]["result"] == "fail"
        assert checks(report)["budget_eur"]["result"] == "requires_review"


def test_nested_values_and_partial_validation(client, monkeypatch):
    data = definition(client)
    data["answer_json"] = {"result": {"budget": "number", "unknown": "boolean"}, "notes": "text"}
    data["validation_json"] = {"result": {"budget": 2800000}}
    model(monkeypatch, raw='{"result":{"budget":2800000.0,"unknown":false},"notes":"Not assessed"}')
    result = checks(execute(client, data)[0])
    assert set(result) == {"answer_json", "result.budget"}
    assert all(item["result"] == "pass" for item in result.values())


def test_empty_expected_fields_are_ignored_and_excluded_from_chart(client, monkeypatch):
    data = definition(client)
    data["validation_json"] = {"budget_eur": "", "supporting_source_id": ""}
    model(monkeypatch, budget=1, version="1")
    report = execute(client, data)[0]
    result = checks(report)
    assert report["evaluation"]["automatic"]["evaluator_version"] == "defined-json-v2"
    for field in ("budget_eur", "supporting_source_id"):
        assert result[field]["result"] == "not_evaluated"
        assert result[field]["applicability"] == "not_applicable"
        assert result[field]["method"] == "ignored"
    summary = client.get(f"/api/batches/{report['run']['batch_id']}").json()
    assert [item["id"] for item in summary["property_counts"]] == ["answer_json"]


def test_nested_ignored_fields_and_null_remain_distinct(client, monkeypatch):
    data = definition(client)
    data["answer_json"] = {
        "result": {"budget": "number", "note": "text"},
        "items": ["text"],
        "unknown": "value or null",
    }
    data["validation_json"] = {"result": {"budget": 2800000, "note": ""}, "items": [""], "unknown": None}
    model(
        monkeypatch, raw='{"result":{"budget":2800000,"note":"anything"},"items":["anything"],"unknown":null}'
    )
    result = checks(execute(client, data)[0])
    assert result["result.budget"]["result"] == "pass"
    assert result["result.note"]["method"] == "ignored"
    assert result["items"]["method"] == "ignored"
    assert result["unknown"]["result"] == "pass"


@pytest.mark.parametrize(
    "context",
    [
        "Output:\n{{output_json}}",
        "Output {{output_json}}\nAgain {{output_json}}",
        "Question: {{user_question}}\nOutput: {{output_json}}",
        "Question: {{user_question}}\nAgain: {{user_question}}",
        "Leave my instructions unchanged.",
    ],
)
def test_answer_json_slot_matches_preview_and_saved_request(client, monkeypatch, context):
    data = definition(client)
    data["context"] = context
    model(monkeypatch)
    preview = client.post(
        "/api/context-preview",
        json={
            "question": data["question"],
            "version_ids": data["version_ids"],
            "context_mode": "explicit",
            "context_ids": [],
            "prompt_instructions": context,
            "answer_json": data["answer_json"],
        },
    ).json()
    rendered = context.replace("{{user_question}}", data["question"]).replace(
        "{{output_json}}", json.dumps(data["answer_json"], ensure_ascii=False, indent=2)
    )
    assert preview["system_prompt"].endswith(rendered)
    direct = execute(client, data)[0]
    frozen = execute(
        client,
        data,
        system_prompt=preview["system_prompt"],
        llm_context=preview["text"],
        context_chunk_ids=preview["chunk_ids"],
    )[0]
    for report in (direct, frozen):
        run = report["run"]
        assert run["messages"][0]["content"] == preview["system_prompt"]
        assert run["snapshot"]["experiment"]["definition"]["context"] == context
        assert not run["snapshot"]["system_prompt_edited"]
        assert ("question-slot-v1" in run["snapshot"]["prompt_version"]) == ("{{user_question}}" in context)
        assert ("answer-json-slot-v1" in run["snapshot"]["prompt_version"]) == ("{{output_json}}" in context)
        assert "2800000" not in json.dumps(run["snapshot"]["request_body"])
    if "{{output_json}}" not in context:
        assert "number or null" not in preview["system_prompt"]


def test_demo_and_truncation_never_pass(client, monkeypatch):
    data = definition(client)
    response = client.post("/api/experiment-definitions/run", json={"definition": data})
    identity = response.json()["run_ids"][0]
    report = client.get(f"/api/runs/{identity}/report").json()
    assert all(item["result"] == "requires_review" for item in checks(report).values())
    model(monkeypatch, finish="length")
    assert all(item["result"] == "requires_review" for item in checks(execute(client, data)[0]).values())


def test_invalid_definition_creates_no_runs(client):
    data = definition(client)
    for change in [
        {"validation_json": {"unknown_field": 1}},
        {"validation_json": {"budget_eur": {"$source": {"version": "2"}}}},
        {"version_ids": [data["version_ids"][0]] * 2},
        {"version_ids": ["nonexistent"]},
        {"title": " "},
    ]:
        response = client.post("/api/experiment-definitions/run", json={"definition": {**data, **change}})
        assert response.status_code == 422, response.text
    assert client.get("/api/runs").json() == []


def test_edited_context_preserves_exact_text_without_false_source_credit(client, monkeypatch):
    data = definition(client)
    model(monkeypatch, raw='{"budget_eur":2800000,"supporting_source_id":"[unknown]"}')
    # This double intentionally accepts the empty verified-source list for edited input.
    monkeypatch.setattr(
        "traceability_lab.pipeline.generate",
        lambda *args: {
            "choices": [
                {
                    "message": {"content": '{"budget_eur":2800000,"supporting_source_id":"[unknown]"}'},
                    "finish_reason": "stop",
                }
            ]
        },
    )
    report = execute(client, data, llm_context="Researcher edited context", context_chunk_ids=[])[0]
    assert report["run"]["messages"][-1]["content"].startswith("Researcher edited context")
    result = checks(report)
    assert result["budget_eur"]["result"] == "pass"
    assert result["supporting_source_id"]["result"] == "requires_review"


def test_sources_slot_is_one_context_in_preview_and_execution(client, monkeypatch):
    data = definition(client)
    data["context"] = "Question: {{user_question}}\nEvidence:\n{{documents}}\nOutput: {{output_json}}"
    model(monkeypatch)
    preview = client.post(
        "/api/context-preview",
        json={
            "question": data["question"],
            "version_ids": data["version_ids"],
            "context_mode": "explicit",
            "context_ids": [],
            "prompt_instructions": data["context"],
            "answer_json": data["answer_json"],
        },
    ).json()
    assert preview["text"].startswith("Question: " + data["question"])
    assert "Document: orion;" in preview["text"]
    assert "{{documents}}" not in preview["text"]
    assert "EXPERIMENT INSTRUCTIONS" not in preview["system_prompt"]
    for overrides in (
        {},
        {
            "system_prompt": preview["system_prompt"],
            "llm_context": preview["text"],
            "context_chunk_ids": preview["chunk_ids"],
        },
    ):
        run = execute(client, data, **overrides)[0]["run"]
        assert run["messages"] == [
            {"role": "system", "content": preview["system_prompt"]},
            {"role": "user", "content": preview["text"]},
        ]
        assert not run["snapshot"]["context_edited"]
        assert run["snapshot"]["experiment"]["definition"]["context"] == data["context"]


def test_experiment_system_prompt_is_saved_previewed_and_sent(client, monkeypatch):
    data = definition(client)
    data["system_prompt"] = "Custom researcher system policy."
    data["context"] = "{{user_question}}\n{{documents}}\n{{output_json}}"
    saved = client.post("/api/experiment-definitions", json=data).json()
    assert saved["definition"]["system_prompt"] == data["system_prompt"]
    preview = client.post(
        "/api/context-preview",
        json={
            "question": data["question"],
            "version_ids": data["version_ids"],
            "context_mode": "explicit",
            "context_ids": [],
            "system_prompt": data["system_prompt"],
            "prompt_instructions": data["context"],
            "answer_json": data["answer_json"],
        },
    ).json()
    assert preview["system_prompt"] == data["system_prompt"]
    model(monkeypatch)
    report = execute(client, data, saved_id=saved["id"])[0]
    assert report["run"]["messages"][0]["content"] == data["system_prompt"]
    assert report["run"]["messages"][1]["content"] == preview["text"]


@pytest.mark.parametrize("bracketed", [False, True])
def test_literal_source_id_validation_does_not_normalize_brackets(client, monkeypatch, bracketed):
    data = definition(client)
    source = client.post(
        "/api/context-preview",
        json={
            "question": data["question"],
            "version_ids": data["version_ids"],
            "context_mode": "explicit",
            "context_ids": [],
        },
    ).json()["chunk_ids"][0]
    data["validation_json"] = {"supporting_source_id": source}
    actual = f"[{source}]" if bracketed else source
    model(monkeypatch, raw=json.dumps({"budget_eur": 2800000, "supporting_source_id": actual}))
    result = checks(execute(client, data)[0])
    assert result["supporting_source_id"]["result"] == ("fail" if bracketed else "pass")
    assert set(result) == {"answer_json", "supporting_source_id"}


def test_delete_experiment_preserves_frozen_run_and_evaluation(client, monkeypatch):
    data = definition(client)
    saved = client.post("/api/experiment-definitions", json=data).json()
    model(monkeypatch)
    report = execute(client, data, saved_id=saved["id"])[0]
    identity = report["run"]["id"]
    before_run = client.get(f"/api/runs/{identity}").json()
    before_evaluation = client.get(f"/api/runs/{identity}/evaluation").json()
    response = client.delete(f"/api/experiment-definitions/{saved['id']}")
    assert response.status_code == 200
    assert response.json() == {"id": saved["id"], "deleted": True}
    assert client.get("/api/experiment-definitions").json() == []
    assert client.get(f"/api/runs/{identity}").json() == before_run
    assert client.get(f"/api/runs/{identity}/evaluation").json() == before_evaluation
    assert client.delete(f"/api/experiment-definitions/{saved['id']}").status_code == 404


def test_experiment_list_is_alphabetical_instead_of_creation_order(client):
    data = definition(client)
    titles = ["Zulu", "beta", "Alpha"]
    saved = []
    for title in titles:
        response = client.post("/api/experiment-definitions", json={**data, "title": title})
        assert response.status_code == 201, response.text
        saved.append(response.json()["id"])
    listed = [
        item["definition"]["title"]
        for item in client.get("/api/experiment-definitions").json()
        if item["id"] in saved
    ]
    assert listed == ["Alpha", "beta", "Zulu"]
