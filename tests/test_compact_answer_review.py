"""Compact answer requests preserve evidence and separate evaluation failures."""

import json
from copy import deepcopy

import pytest
from test_quoted_facts import quoted_experiment

from traceability_lab.semantic_evaluation import ANSWER_EVALUATOR_POLICY, answer_review_payload


def test_answer_payload_has_only_four_fields_and_all_original_documents():
    run = {
        "question": "Question?",
        "context": [
            {
                "document_id": "a",
                "version": "1",
                "start": 0,
                "end": 3,
                "text": "a\n\n",
                "sha256": "private",
                "chunk_id": "x",
            },
            {"document_id": "a", "version": "1", "start": 3, "end": 4, "text": "b"},
            {"document_id": "uncited", "version": "2", "start": 0, "end": 8, "text": "Evidence"},
        ],
        "snapshot": {
            "evaluation_reference": {
                "definition": {
                    "documents": [
                        {"document_id": "a", "version": "1", "content": "a\n\nb"},
                        {"document_id": "uncited", "version": "2", "content": "Evidence"},
                    ]
                }
            }
        },
    }
    original = deepcopy(run)
    payload = answer_review_payload(
        run,
        {"facts": [], "answer": "Answer only"},
        {"facts": ["PRIVATE REFERENCE FACT"], "answer": "Reference"},
    )
    assert set(payload) == {"question", "expected_answer", "generated_answer", "documents"}
    assert payload["generated_answer"] == "Answer only"
    assert payload["documents"] == [
        {"document_id": "a", "version": "1", "text": "a\n\nb"},
        {"document_id": "uncited", "version": "2", "text": "Evidence"},
    ]
    assert "PRIVATE REFERENCE FACT" not in json.dumps(payload)
    assert run == original
    run["context"][1]["start"] = 4
    with pytest.raises(ValueError, match="contiguous"):
        answer_review_payload(run, {"answer": "Answer"}, None)


@pytest.mark.parametrize(
    "expected,generated,meaning,support,unsupported,score,result",
    [
        (
            "Approval is pending.",
            "Approval has not yet been granted.",
            "equivalent",
            "supported",
            False,
            100,
            "pass",
        ),
        (
            "Only a proposal exists.",
            "The proposed amount is approved.",
            "not_equivalent",
            "contradicted",
            True,
            0,
            "fail",
        ),
        (
            "Amount and date are established.",
            "The amount is established.",
            "partial",
            "supported",
            False,
            60,
            "fail",
        ),
        (
            "No recorded approval in this document.",
            "No approval exists anywhere.",
            "not_equivalent",
            "insufficient_evidence",
            True,
            0,
            "fail",
        ),
        (None, "The documents do not determine a value.", "equivalent", "supported", False, 100, "pass"),
        (
            "The documents conflict about the amount.",
            None,
            "not_equivalent",
            "insufficient_evidence",
            False,
            0,
            "fail",
        ),
        (
            "The sources conflict; the amount is ambiguous.",
            "Conflicting documents prevent an unambiguous amount.",
            "equivalent",
            "supported",
            False,
            100,
            "pass",
        ),
    ],
)
def test_answer_contract_and_outcomes(
    client, monkeypatch, expected, generated, meaning, support, unsupported, score, result
):
    definition = quoted_experiment()
    definition["semantic_reference"] = {"facts": [], "answer": expected}
    answer = {"facts": [], "answer": generated}
    monkeypatch.setattr(
        "traceability_lab.pipeline.generate",
        lambda *args: {"choices": [{"message": {"content": json.dumps(answer)}, "finish_reason": "stop"}]},
    )
    calls = []

    def reviewer(settings, body, chunks):
        payload = json.loads(body["messages"][1]["content"])
        calls.append(body)
        assert set(payload) == {"question", "expected_answer", "generated_answer", "documents"}
        assert payload["expected_answer"] == expected
        assert payload["generated_answer"] == generated
        assert {doc["document_id"] for doc in payload["documents"]} == {"orion", "atlas"}
        assert all(set(doc) == {"document_id", "version", "text"} for doc in payload["documents"])
        value = {
            "answer": {
                "meaning_match": meaning,
                "evidence_support": support,
                "confidence": 100,
                "correctness_score": score,
                "explanation": "Controlled test judgment.",
            },
            "unsupported_answer": unsupported,
        }
        return {"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]}

    monkeypatch.setattr("traceability_lab.semantic_evaluation.generate", reviewer)
    response = client.post(
        "/api/experiment-definitions/run",
        json={"definition": definition, "execution": {"repetitions": 1, "llm_profile_id": "test-generator"}},
    )
    assert response.status_code == 201, response.text
    batch = client.get(f"/api/batches/{response.json()['batch_id']}").json()
    criterion = next(item for item in batch["evaluations"][0]["result_criteria"] if item["id"] == "answer")
    assert criterion["result"] == result
    assert len(calls) == 1


@pytest.mark.parametrize("raw", ["not json", '{"answer":{"meaning_match":"equivalent"},"unexpected":true}'])
def test_invalid_answer_review_is_an_evaluation_error(client, monkeypatch, raw):
    definition = quoted_experiment()
    definition["semantic_reference"] = {"facts": [], "answer": "Reference"}
    monkeypatch.setattr(
        "traceability_lab.pipeline.generate",
        lambda *args: {
            "choices": [
                {"message": {"content": '{"facts":[],"answer":"Different wording"}'}, "finish_reason": "stop"}
            ]
        },
    )
    monkeypatch.setattr(
        "traceability_lab.semantic_evaluation.generate",
        lambda *args: {"choices": [{"message": {"content": raw}, "finish_reason": "stop"}]},
    )
    response = client.post(
        "/api/experiment-definitions/run",
        json={"definition": definition, "execution": {"repetitions": 1, "llm_profile_id": "test-generator"}},
    )
    assert response.status_code == 201, response.text
    batch = client.get(f"/api/batches/{response.json()['batch_id']}").json()
    criterion = next(item for item in batch["evaluations"][0]["result_criteria"] if item["id"] == "answer")
    assert criterion["result"] == "evaluation_error"
    counts = next(item for item in batch["property_counts"] if item["id"] == "answer")
    assert counts["evaluation_error"] == 1 and counts["fail"] == 0


def test_prompt_is_generic_compact_and_explicit_about_scope():
    assert len(ANSWER_EVALUATOR_POLICY) < 1400
    assert "Orion" not in ANSWER_EVALUATOR_POLICY
    assert "Do not treat missing evidence as proof that something does not exist" in ANSWER_EVALUATOR_POLICY
    assert "conflicting evidence" in ANSWER_EVALUATOR_POLICY
    assert "Never assume missing qualifications or context" in ANSWER_EVALUATOR_POLICY


@pytest.mark.parametrize("observed", [True, False])
def test_schema_enforcement_only_for_observed_llama_server(client, monkeypatch, observed):
    from test_quoted_facts import configure_calls, execute

    import traceability_lab.pipeline as pipeline

    calls = configure_calls(monkeypatch)
    original = pipeline.capture_identity

    def identity(*args, **kwargs):
        result = original(*args, **kwargs)
        result["runtime"].update(
            engine="llama.cpp",
            field_sources={"engine": "server_observed" if observed else "deployment_declared"},
        )
        return result

    monkeypatch.setattr(pipeline, "capture_identity", identity)
    _, observation = execute(client)
    body = calls[-1]
    assert ("response_format" in body) is observed
    assert ("Format:" in body["messages"][0]["content"]) is (not observed)
    assert observation["answer_prompt_version"] == "answer-semantic-v3"
    assert "output_schema" not in json.loads(body["messages"][1]["content"])
    assert all("response_format" not in call for call in calls[:-1])
    if observed:
        assert body["response_format"]["schema"]["additionalProperties"] is False
        assert "correctness_score" in body["response_format"]["schema"]["$defs"]["AnswerJudgment"]["required"]
    assert observation["calls"][-1]["output_enforcement"] == ("json_schema" if observed else "compact_prompt")


def test_extra_fact_field_is_error_but_answer_still_runs(client, monkeypatch):
    from test_quoted_facts import configure_calls, execute

    import traceability_lab.semantic_evaluation as semantic

    configure_calls(monkeypatch)
    original = semantic.generate

    def reviewer(settings, body, chunks):
        response = original(settings, body, chunks)
        payload = json.loads(body["messages"][1]["content"])
        if payload.get("fact_index") == 0:
            value = json.loads(response["choices"][0]["message"]["content"])
            value["source_support_detail"] = "Forbidden extra field"
            response["choices"][0]["message"]["content"] = json.dumps(value)
        return response

    monkeypatch.setattr(semantic, "generate", reviewer)
    batch, observation = execute(client)
    assert observation["answer_review_completed"]
    assert observation["calls"][0]["status"] == "failed"
    assert "source_support_detail" in observation["calls"][0]["raw_answer"]
    assert observation["calls"][-1]["status"] == "completed"
    criteria = {item["id"]: item for item in batch["evaluations"][0]["result_criteria"]}
    assert criteria["facts"]["result"] == "evaluation_error"
    assert criteria["answer"]["result"] == "pass"
