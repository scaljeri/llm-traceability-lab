"""New answer contract, legacy definitions and immutable historical display."""

import copy
import json

from test_quoted_facts import configure_calls, execute, quoted_experiment

from traceability_lab.fact_answers import FACT_ANSWER_JSON, fact_result_criteria, valid_fact_answer
from traceability_lab.models import SemanticReference


def test_answer_is_required_and_old_generated_field_is_rejected():
    assert "answer" in FACT_ANSWER_JSON and "conclusion" not in FACT_ANSWER_JSON
    assert valid_fact_answer({"facts": [], "answer": None})
    assert not valid_fact_answer({"facts": [], "conclusion": None})
    assert SemanticReference.model_validate({"conclusion": "Legacy reference"}).model_dump() == {
        "facts": [],
        "answer": "Legacy reference",
    }


def test_legacy_saved_definition_is_exposed_and_exported_with_answer(client, db):
    definition = quoted_experiment()
    definition["output_mode"] = "facts_and_conclusion"
    reference = definition["semantic_reference"]
    reference["conclusion"] = reference.pop("answer")
    from traceability_lab.db import json_text, now

    with db.connect() as connection:
        connection.execute(
            "INSERT INTO experiment_definitions VALUES (?,?,?,?,?)",
            ("legacy-answer", 1, json_text(definition), now(), now()),
        )
    item = next(p for p in client.get("/api/experiment-definitions").json() if p["id"] == "legacy-answer")
    assert item["definition"]["semantic_reference"]["answer"] == reference["conclusion"]
    assert "conclusion" not in item["definition"]["semantic_reference"]
    assert (
        json.loads(
            db.one("SELECT definition FROM experiment_definitions WHERE id=?", ("legacy-answer",))[
                "definition"
            ]
        )
        == definition
    )


def test_historical_judgments_remain_readable_without_mutating_raw_data(client, monkeypatch):
    configure_calls(monkeypatch)
    batch, _ = execute(client)
    run = client.get(f"/api/runs/{batch['runs'][0]['id']}").json()
    evaluation = batch["evaluations"][0]
    expected = fact_result_criteria(run, evaluation["automatic"])

    def legacy(value):
        if isinstance(value, list):
            return [legacy(item) for item in value]
        if isinstance(value, dict):
            return {
                {"answer": "conclusion", "unsupported_answer": "unsupported_conclusion"}.get(
                    key, key
                ): "conclusion" if key in ("id", "phase") and item == "answer" else legacy(item)
                for key, item in value.items()
            }
        return value

    # Keep the outer run's raw answer field; rename only the generated object and snapshots.
    old = {
        **run,
        "snapshot": legacy(run["snapshot"]),
        "answer": json.dumps(legacy(json.loads(run["answer"]))),
    }
    automatic = legacy(evaluation["automatic"])
    original = copy.deepcopy((old, automatic))
    actual = fact_result_criteria(old, automatic)
    assert [(p["id"], p["result"]) for p in actual] == [(p["id"], p["result"]) for p in expected]
    assert (old, automatic) == original


def test_all_data_experiments_import_with_answer(client):
    from pathlib import Path

    for path in sorted(Path("experiments").glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("format") != "traceability-experiment-v2":
            continue
        response = client.post("/api/experiment-definitions/import", json=payload)
        assert response.status_code == 200, (path, response.text)
        definition = response.json()["definition"]
        assert definition["output_mode"] == "facts_and_answer"
        assert "answer" in definition["answer_json"] and "conclusion" not in definition["answer_json"]
        assert "answer" in definition["semantic_reference"]
