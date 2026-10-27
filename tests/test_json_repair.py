"""A format-only second generator request is explicit and never sees expected answers."""

import json
from copy import deepcopy

import pytest
from test_quoted_facts import quoted_experiment


@pytest.mark.parametrize("first", ['{"facts":[],"answer":null', '{"fact":"A statement","sources":[]}'])
def test_invalid_json_gets_one_recorded_repair_with_original_context(client, monkeypatch, first):
    calls = []
    final = '{"facts":[],"answer":null}'
    responses = [first, final]

    def model(settings, body, chunks):
        calls.append(deepcopy(body))
        return {
            "choices": [{"message": {"content": responses[len(calls) - 1]}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    monkeypatch.setattr("traceability_lab.pipeline.generate", model)
    definition = quoted_experiment()
    definition["semantic_reference"] = {"facts": [], "answer": None}
    definition["semantic_evaluator"] = None
    result = client.post(
        "/api/experiment-definitions/run",
        json={
            "definition": definition,
            "execution": {"repetitions": 1, "llm_profile_id": "test-generator", "temperature": 1.2},
        },
    )
    assert result.status_code == 201, result.text
    batch = client.get(f"/api/batches/{result.json()['batch_id']}").json()
    run = client.get(f"/api/runs/{batch['runs'][0]['id']}").json()
    assert len(calls) == 2
    assert calls[1]["messages"][:-2] == calls[0]["messages"]
    assert calls[1]["messages"][-2] == {"role": "assistant", "content": first}
    assert "only" in calls[1]["messages"][-1]["content"].lower()
    assert "semantic_reference" not in json.dumps(calls[1])
    assert calls[0]["temperature"] == calls[1]["temperature"] == 1.2
    assert run["answer"] == final
    repair = run["snapshot"]["json_repair"]
    assert repair["original_answer"] == first
    assert repair["request_body"] == calls[1]
    assert repair["response"]["choices"][0]["message"]["content"] == final
    assert repair["status"] == "repaired"
    assert batch["overview"]["generator"]["tokens"]["calls"] == 2
    assert batch["overview"]["generator"]["tokens"]["input_tokens"] == 20
    assert batch["overview"]["generator"]["tokens"]["output_tokens"] == 10
    assert batch["overview"]["generator"]["json_format"] == {
        "checked": 1,
        "initial_valid": 0,
        "repair_attempts": 1,
        "repaired": 1,
    }
    assert batch["runs"][0]["json_repair_status"] == "repaired"


def test_valid_json_never_calls_repair(client, monkeypatch):
    calls = []

    def model(*args):
        calls.append(args)
        return {"choices": [{"message": {"content": '{"facts":[],"answer":null}'}, "finish_reason": "stop"}]}

    monkeypatch.setattr("traceability_lab.pipeline.generate", model)
    definition = quoted_experiment()
    definition["semantic_evaluator"] = None
    result = client.post(
        "/api/experiment-definitions/run",
        json={"definition": definition, "execution": {"repetitions": 1, "llm_profile_id": "test-generator"}},
    )
    assert result.status_code == 201
    assert len(calls) == 1


@pytest.mark.parametrize("repair_result", ["still invalid", None])
def test_repair_is_bounded_and_failed_attempt_keeps_both_outputs(client, monkeypatch, repair_result):
    calls = []
    first = '{"facts":[],"answer":null'

    def model(*args):
        calls.append(args)
        assert len(calls) <= 2
        if len(calls) == 2 and repair_result is None:
            raise RuntimeError("Repair server unavailable")
        answer = first if len(calls) == 1 else repair_result
        return {"choices": [{"message": {"content": answer}, "finish_reason": "stop"}]}

    monkeypatch.setattr("traceability_lab.pipeline.generate", model)
    definition = quoted_experiment()
    definition["semantic_evaluator"] = None
    response = client.post(
        "/api/experiment-definitions/run",
        json={"definition": definition, "execution": {"repetitions": 1, "llm_profile_id": "test-generator"}},
    )
    batch = client.get(f"/api/batches/{response.json()['batch_id']}").json()
    run = client.get(f"/api/runs/{batch['runs'][0]['id']}").json()
    assert len(calls) == 2
    assert run["answer"] == (first if repair_result is None else repair_result)
    repair = run["snapshot"]["json_repair"]
    assert repair["original_answer"] == first
    assert repair["status"] == ("error" if repair_result is None else "failed")
    assert all(item["fail"] == 1 for item in batch["property_counts"])


def test_cancel_interrupts_json_repair_without_losing_original_output(client, monkeypatch):
    import threading

    from test_active_executions import wait_for

    from traceability_lab.cancellation import cancellation_event, check_cancelled

    entered = threading.Event()
    calls = []
    first = '{"facts":[],"answer":null'

    def model(*args):
        calls.append(args)
        if len(calls) == 2:
            entered.set()
            assert cancellation_event().wait(3)
            check_cancelled()
        return {"choices": [{"message": {"content": first}, "finish_reason": "stop"}]}

    monkeypatch.setattr("traceability_lab.pipeline.generate", model)
    definition = quoted_experiment()
    definition["semantic_evaluator"] = None
    response = client.post(
        "/api/executions",
        json={"definition": definition, "execution": {"llm_profile_id": "test-generator", "repetitions": 3}},
    )
    identity = response.json()["batch_id"]
    assert entered.wait(3)
    client.post(f"/api/executions/{identity}/cancel")
    final = wait_for(client, identity, "cancelled")
    assert final["completed"] == 0 and final["summary"] is None
    assert len(calls) == 2
    row = next(run for run in client.get("/api/runs").json() if run["batch_id"] == identity)
    run = client.get(f"/api/runs/{row['id']}").json()
    assert run["snapshot"]["json_repair"]["status"] == "cancelled"
    assert run["snapshot"]["json_repair"]["original_answer"] == first
    assert run["answer"] == first


def test_empty_custom_json_contract_still_checks_syntax(client, monkeypatch):
    calls = []

    def model(*args):
        calls.append(args)
        return {
            "choices": [
                {"message": {"content": "{broken" if len(calls) == 1 else "{}"}, "finish_reason": "stop"}
            ]
        }

    monkeypatch.setattr("traceability_lab.pipeline.generate", model)
    definition = {
        "title": "Empty JSON object",
        "question": "Return an empty JSON object.",
        "documents": [],
        "output_mode": "custom",
        "answer_json": {},
        "validation_json": {},
        "context": "Return one JSON object.",
    }
    response = client.post(
        "/api/experiment-definitions/run",
        json={"definition": definition, "execution": {"repetitions": 1, "llm_profile_id": "test-generator"}},
    )
    assert response.status_code == 201, response.text
    batch = client.get(f"/api/batches/{response.json()['batch_id']}").json()
    assert len(calls) == 2
    assert batch["runs"][0]["answer"] == "{}"
    assert batch["runs"][0]["json_repair_status"] == "repaired"
