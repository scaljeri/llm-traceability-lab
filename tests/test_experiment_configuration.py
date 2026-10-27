"""Scenario variants and frozen evaluations; all model outputs are test doubles."""

import json
import sqlite3

import pytest

from traceability_lab.experiment_evaluation import parse_answer


def source_ids(client):
    return {
        doc["version"]: doc["id"]
        for doc in client.get("/api/documents").json()
        if doc["document_id"] == "orion"
    }


def run_version(client, **overrides):
    response = client.post("/test-scenarios/version/run", json={"context_ids": [], **overrides})
    assert response.status_code == 201, response.text
    return client.get("/api/runs/" + response.json()["run_ids"][0]).json()


def criteria(client, run):
    result = client.get(f"/api/runs/{run['id']}/evaluation").json()
    return {item["id"]: item for item in result["automatic"]["criteria"]}


def model_double(
    monkeypatch, *, budget=2800000, citation="2", reversed_relation=False, raw=None, finish="stop"
):
    def generate(settings, body, chunks):
        assert settings.backend == "local"
        by_version = {item["version"]: item["id"] for item in chunks}
        answer = raw
        if answer is None:
            answer = json.dumps(
                {
                    "budget_eur": budget,
                    "supporting_source_id": f"[{by_version.get(citation, 'nonexistent')}]",
                    "version_relation": {
                        "older_source_id": f"[{by_version.get('2' if reversed_relation else '1', 'absent')}]",
                        "newer_source_id": f"[{by_version.get('1' if reversed_relation else '2', 'absent')}]",
                    },
                    "explanation": "Deterministic test double, not an observed model finding.",
                }
            )
        return {"choices": [{"message": {"content": answer}, "finish_reason": finish}]}

    monkeypatch.setattr("traceability_lab.pipeline.generate", generate)


def test_default_version_scenario_and_order_override(client, monkeypatch):
    monkeypatch.setattr(
        "traceability_lab.pipeline.retrieve", lambda *args: pytest.fail("Unexpected retrieval")
    )
    ids = source_ids(client)
    first = run_version(client)
    second = run_version(client, version_ids=[ids["2"], ids["1"]])
    assert first["question"] == "What is the current budget for Project Orion?"
    assert [item["version"] for item in first["context"]] == ["1", "2"]
    assert [item["version"] for item in second["context"]] == ["2", "1"]
    assert first["retrievals"] == second["retrievals"] == []
    assert not first["snapshot"]["experiment"]["modified"]
    assert second["snapshot"]["experiment"]["overrides"] == {"versions": [["orion", "2"], ["orion", "1"]]}
    assert first["messages"][0] == second["messages"][0]
    assert "higher document version" in first["messages"][0]["content"]
    assert "2800000" not in json.dumps(first["messages"])
    assert "approved_sources" not in json.dumps(first["messages"])
    assert first["snapshot"]["experiment"]["definition_version"] == "3"


@pytest.mark.parametrize("reverse", [False, True])
def test_expected_budget_citation_and_reported_relationship(client, monkeypatch, reverse):
    model_double(monkeypatch)
    ids = source_ids(client)
    run = run_version(
        client,
        llm_profile_id="test-generator",
        version_ids=[ids[v] for v in (["2", "1"] if reverse else ["1", "2"])],
    )
    result = criteria(client, run)
    for key in ["budget", "supporting_version", "version_relation"]:
        assert result[key]["applicability"] == "applicable"
        assert result[key]["result"] == "pass"
    assert result["unsupported_claims"]["result"] == "requires_review"
    assert result["retrieval_evidence"]["applicability"] == "not_applicable"
    assert result["retrieval_evidence"]["result"] == "not_evaluated"


@pytest.mark.parametrize(
    ("budget", "citation", "relation", "failed"),
    [
        (2400000, "2", False, "budget"),
        (2800000, "1", False, "supporting_version"),
        (2800000, "unknown", False, "supporting_version"),
        (2800000, "2", True, "version_relation"),
    ],
)
def test_independent_wrong_budget_citation_and_relation(
    client, monkeypatch, budget, citation, relation, failed
):
    model_double(monkeypatch, budget=budget, citation=citation, reversed_relation=relation)
    run = run_version(client, llm_profile_id="test-generator")
    result = criteria(client, run)
    assert result[failed]["result"] == "fail"
    assert all(
        result[key]["result"] == "pass"
        for key in ["budget", "supporting_version", "version_relation"]
        if key != failed
    )


@pytest.mark.parametrize(
    "raw",
    [
        "Het budget is 2,8 miljoen euro.",
        '{"budget_eur":2800000,"budget_eur":2400000}',
        '{"budget_eur":true}',
        '{"budget_eur":NaN}',
        '{"budget_eur":null}',
    ],
)
def test_ambiguous_or_unstructured_answer_requires_review(client, monkeypatch, raw):
    model_double(monkeypatch, raw=raw)
    run = run_version(client, llm_profile_id="test-generator")
    assert criteria(client, run)["budget"]["result"] == "requires_review"


def test_missing_required_source_is_not_applicable(client, monkeypatch):
    model_double(monkeypatch)
    run = run_version(client, llm_profile_id="test-generator", version_ids=[source_ids(client)["1"]])
    result = criteria(client, run)
    for key in ["budget", "supporting_version", "version_relation"]:
        assert result[key]["applicability"] == "not_applicable"
        assert result[key]["result"] == "not_evaluated"
        assert result[key]["reason"]


def test_changed_instructions_make_semantic_applicability_undetermined(client, monkeypatch):
    model_double(monkeypatch)
    run = run_version(
        client, llm_profile_id="test-generator", prompt_instructions="Prefer the earlier version."
    )
    assert run["snapshot"]["experiment"]["modified"]
    assert "Prefer the earlier version." in run["messages"][0]["content"]
    for key in ["budget", "supporting_version", "version_relation"]:
        criterion = criteria(client, run)[key]
        assert criterion["applicability"] == "undetermined"
        assert criterion["result"] == "not_evaluated"


def test_retrieval_override_uses_actual_context_for_applicability(client, monkeypatch):
    model_double(monkeypatch)
    monkeypatch.setattr(
        "traceability_lab.embeddings.Embedder.encode", lambda *args, **kwargs: [1.0] + [0.0] * 1023
    )
    # A deterministic retrieval double supplies only v1, despite both configured versions.
    monkeypatch.setattr("traceability_lab.pipeline.ensure_embeddings", lambda *args: None)
    monkeypatch.setattr(
        "traceability_lab.pipeline.retrieve",
        lambda db, chunks, *args: [
            {
                **next(c for c in chunks if c["version"] == "1"),
                "rank": 1,
                "distance": 0.1,
                "metric": "cosine_distance",
            }
        ],
    )
    run = run_version(
        client,
        llm_profile_id="test-generator",
        embedding_profile_id="test-embedding",
        context_mode="retrieval",
        top_k=1,
    )
    assert len(run["snapshot"]["corpus"]) == 2 and len(run["context"]) == 1
    assert run["snapshot"]["experiment"]["overrides"]["context_mode"] == "retrieval"
    assert criteria(client, run)["budget"]["applicability"] == "not_applicable"


def test_frozen_reference_and_evaluation_survive_reference_edits(client, monkeypatch, tmp_path, db):
    model_double(monkeypatch)
    run = run_version(client, llm_profile_id="test-generator")
    path = f"/api/runs/{run['id']}"
    before = client.get(path + "/report").json()
    replacement = tmp_path / "different-ground-truth.json"
    replacement.write_text(json.dumps({"cases": {"version": {"expected": "Changed", "budget_eur": 999}}}))
    monkeypatch.setattr("scenario_fixtures.REFERENCE", replacement)
    assert client.get(path + "/report").json() == before
    assert before["evaluation_reference"]["budget_eur"] == 2800000
    assert before["evaluation"]["automatic"]["trace_id"] == run["trace_id"]
    assert before["evaluation"]["automatic"]["provenance_uri"].endswith("/provenance")
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with db.connect() as connection:
            connection.execute("UPDATE run_evaluations SET payload=? WHERE run_id=?", ("{}", run["id"]))


@pytest.mark.parametrize(
    "payload", [{"version_ids": []}, {"version_ids": ["unknown"]}, {"context_mode": "bad"}, {"top_k": 0}]
)
def test_invalid_overrides_do_not_create_runs(client, payload):
    response = client.post("/test-scenarios/version/run", json=payload)
    assert response.status_code in {404, 422}
    assert client.get("/api/runs").json() == []


def test_demo_never_passes_answer_criteria(client):
    run = run_version(client)
    assert all(item["result"] != "pass" for item in criteria(client, run).values())


def test_json_fence_is_supported_but_trailing_prose_is_not():
    assert parse_answer('```json\n{"budget_eur":2800000}\n```') == {"budget_eur": 2800000}
    assert parse_answer('{"budget_eur":2800000} and more claims') is None


def test_changed_document_body_does_not_inherit_a_label_based_expectation(client, monkeypatch):
    model_double(monkeypatch)
    ids = source_ids(client)
    assert client.delete("/api/document-versions/" + ids["2"]).status_code == 200
    replacement = client.post(
        "/api/documents",
        json={
            "document_id": "orion",
            "version": "2",
            "filename": "changed.txt",
            "body": "A changed project card: budget is 9 million.",
        },
    ).json()
    run = run_version(client, llm_profile_id="test-generator", version_ids=[ids["1"], replacement["id"]])
    result = criteria(client, run)
    assert result["budget"]["applicability"] == "undetermined"
    assert result["budget"]["result"] == "not_evaluated"
    assert "changed source" in result["budget"]["reason"]


def test_truncated_structured_answer_does_not_pass(client, monkeypatch):
    model_double(monkeypatch, finish="length")
    run = run_version(client, llm_profile_id="test-generator")
    assert criteria(client, run)["budget"]["result"] == "requires_review"


def test_additional_operator_instructions_are_frozen_and_require_applicability_review(client, monkeypatch):
    model_double(monkeypatch)
    context = client.post(
        "/api/contexts", json={"title": "Variant", "content": "Use a different budget.", "active": True}
    ).json()
    run = run_version(client, llm_profile_id="test-generator", context_ids=[context["id"]])
    assert run["snapshot"]["experiment"]["overrides"]["context_ids"] == [context["id"]]
    assert criteria(client, run)["budget"]["applicability"] == "undetermined"
    assert run["snapshot"]["supplemental_contexts"][0]["content"] == "Use a different budget."


def test_old_runs_without_a_frozen_reference_are_not_graded_against_current_truth(client, db):
    from traceability_lab.db import json_text

    with db.connect() as connection:
        connection.execute(
            "INSERT INTO runs (id,created_at,status,trace_id,question,mode,backend,case_id,snapshot) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "legacy",
                "2020",
                "completed",
                "old-trace",
                "Old?",
                "strict",
                "demo",
                "version",
                json_text({"corpus": [], "required_spans": []}),
            ),
        )
    result = client.get("/api/runs/legacy/evaluation").json()
    assert result["reference_status"] == "not_recorded"
    assert result["reference"] is None and result["automatic"] is None
    assert result["retrieval_recall_at_k"]["value"] is None


def test_budget_can_apply_without_older_version_but_relationship_cannot(client, monkeypatch):
    model_double(monkeypatch)
    run = run_version(client, llm_profile_id="test-generator", version_ids=[source_ids(client)["2"]])
    result = criteria(client, run)
    assert result["budget"]["result"] == "pass"
    assert result["supporting_version"]["result"] == "pass"
    assert result["version_relation"]["applicability"] == "not_applicable"
    assert result["version_relation"]["result"] == "not_evaluated"


def test_omitted_overrides_use_scenario_defaults_not_request_model_defaults(client, monkeypatch):
    from scenario_fixtures import case_by_id

    definition = case_by_id("version")
    definition.update(temperature=0.3, max_tokens=700)
    monkeypatch.setattr("scenario_fixtures.case_by_id", lambda case_id: definition)
    run = run_version(client)
    assert run["snapshot"]["question_configuration"]["temperature"] == 0.3
    assert run["snapshot"]["question_configuration"]["max_tokens"] == 700
    assert not run["snapshot"]["experiment"]["modified"]


def test_edited_question_is_sent_and_frozen_exactly(client, monkeypatch):
    model_double(monkeypatch)
    question = "  Wat vermeldt versie 1?\nBehoud dit exact: € 👩🏽‍🔬  "
    run = run_version(client, llm_profile_id="test-generator", question=question)
    assert run["question"] == question
    assert run["snapshot"]["question_configuration"]["question"] == question
    assert run["messages"][-1]["content"].endswith("QUESTION\n" + question)
    assert run["snapshot"]["request_body"]["messages"] == run["messages"]
    experiment = run["snapshot"]["experiment"]
    assert experiment["modified"]
    assert experiment["overrides"]["question"] == question
    assert experiment["original_defaults"]["question"] == experiment["definition"]["question"]
    for key in ["budget", "supporting_version", "version_relation"]:
        assert criteria(client, run)[key]["applicability"] == "undetermined"
        assert criteria(client, run)[key]["result"] == "not_evaluated"
    report = client.get(f"/api/runs/{run['id']}/report").json()
    assert report["run"]["question"] == question
    assert report["evaluation_reference"]["budget_eur"] == 2800000
    assert any(span["trace_id"] == run["trace_id"] for span in report["trace"])


@pytest.mark.parametrize("question", ["", "   ", "x" * 2001])
def test_invalid_question_override_never_creates_a_run(client, question):
    response = client.post("/test-scenarios/version/run", json={"question": question})
    assert response.status_code == 422
    assert client.get("/api/runs").json() == []


def test_changed_retrieval_question_does_not_reuse_original_reference_score(client):
    response = client.post(
        "/test-scenarios/complete/run", json={"question": "Who manages Orion?", "context_ids": []}
    )
    assert response.status_code == 201
    run_id = response.json()["run_ids"][0]
    result = client.get(f"/api/runs/{run_id}/evaluation").json()
    assert result["retrieval_recall_at_k"] is None
    criterion = result["automatic"]["criteria"][0]
    assert criterion["applicability"] == "undetermined"
    assert criterion["result"] == "not_evaluated"
