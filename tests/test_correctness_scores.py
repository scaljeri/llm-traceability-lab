"""Correctness is a model observation; existing checks and confidence remain authoritative."""

import json
from copy import deepcopy

import pytest
from pydantic import ValidationError
from test_quoted_facts import quoted_answer, quoted_experiment

from traceability_lab.semantic_evaluation import FactJudgment


@pytest.mark.parametrize(
    "case",
    [
        "positive",
        "negative",
        "observational_zero",
        "fabricated",
        "non_contiguous",
        "whitespace",
        "below_confidence",
    ],
)
def test_scores_are_independent_persisted_and_cannot_override_validation(client, monkeypatch, case):
    answer = quoted_answer()
    if case == "fabricated":
        answer["facts"][1]["sources"][0]["quote"] = "Invented quotation."
    elif case == "non_contiguous":
        answer["facts"][1]["sources"][0]["quote"] = "Project Orion ... Project manager: Anna de Vries."

    elif case == "whitespace":
        answer["facts"][1]["sources"][0]["quote"] = answer["facts"][1]["sources"][0]["quote"].replace(
            "\n\n", "\n"
        )

    def generator(settings, body, chunks):
        assert "correctness_score" not in json.dumps(body)
        return {"choices": [{"message": {"content": json.dumps(answer)}, "finish_reason": "stop"}]}

    def evaluator(settings, body, chunks):
        payload = json.loads(body["messages"][1]["content"])
        assert "correctness_score" in body["messages"][0]["content"]
        if "generated_fact" in payload:
            assert "independently of citation validity" in body["messages"][0]["content"]
        else:
            assert {source["document_id"] for source in payload["documents"]} == {"orion", "atlas"}
        if "generated_fact" in payload:
            index = payload["fact_index"]
            value = {
                "fact_index": index,
                "expected_fact_indices": [2 if index == 0 else 0],
                "meaning_match": "equivalent",
                "source_support": "supported",
                "correctness_score": 97,
                "confidence": 100,
                "explanation": "Mocked independent judgment.",
            }
            if case in ("fabricated", "non_contiguous", "whitespace"):
                value["correctness_score"] = 100
            if case == "below_confidence":
                value["confidence"] = 89
            if case in ("negative", "observational_zero"):
                value["correctness_score"] = 0
            if case == "negative":
                value.update(meaning_match="not_equivalent", source_support="contradicted")
        else:
            value = {
                "answer": {
                    "meaning_match": "equivalent",
                    "evidence_support": "supported",
                    "correctness_score": 76,
                    "confidence": 100,
                    "explanation": "Mocked answer judgment.",
                },
                "unsupported_answer": False,
            }
        return {"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]}

    monkeypatch.setattr("traceability_lab.pipeline.generate", generator)
    monkeypatch.setattr("traceability_lab.semantic_evaluation.generate", evaluator)
    definition = quoted_experiment()
    definition["semantic_evaluator"]["confidence_threshold"] = 90
    response = client.post(
        "/api/experiment-definitions/run",
        json={"definition": definition, "execution": {"repetitions": 1, "llm_profile_id": "test-generator"}},
    )
    assert response.status_code == 201, response.text
    batch = client.get(f"/api/batches/{response.json()['batch_id']}").json()
    evaluation = batch["evaluations"][0]
    observation = evaluation["automatic"]["semantic_evaluation"]
    assert observation["status"] == "completed", observation
    assert observation["prompt_version"] == "facts-semantic-v15"
    assert observation["schema_version"] == "semantic-judgments-v3"
    fact = observation["judgments"]["facts"][0]
    assert fact["correctness_score"] == (
        0
        if case in ("negative", "observational_zero")
        else 100
        if case in ("fabricated", "non_contiguous", "whitespace")
        else 97
    )
    assert fact["confidence"] == (89 if case == "below_confidence" else 100)
    assert observation["judgments"]["answer"]["correctness_score"] == 76
    criteria = {item["id"]: item for item in evaluation["result_criteria"]}
    assert criteria["facts"]["result"] == (
        "fail"
        if case in ("negative", "fabricated", "non_contiguous", "whitespace", "below_confidence")
        else "pass"
    )
    assert criteria["answer"]["result"] == "pass"  # 76 is observational; confidence 100 clears threshold 90.
    assert "correctness_score" not in criteria["facts"]  # No aggregate factual-correctness metric.
    if case == "whitespace":
        assert fact["meaning_match"] == "equivalent" and fact["source_support"] == "supported"
        checks = evaluation["automatic"]["criteria"][1]["citation_validations"]
        assert checks[0]["status"] == "pass" and checks[1]["status"] == "fail"
        assert checks[1]["errors"][0]["code"] == "whitespace_mismatch"
    identity = batch["runs"][0]["id"]
    stored = client.get(f"/api/runs/{identity}/evaluation").json()
    assert stored["automatic"]["semantic_evaluation"]["judgments"]["facts"][0] == fact
    report = client.get(f"/api/runs/{identity}/report").json()
    assert "correctness_score" in json.dumps(report)
    trace = client.get(f"/api/runs/{identity}/trace").json()
    span = next(span for span in trace if span["span_id"] == observation["calls"][0]["span_id"])
    assert span["attributes"]["lab.evaluator.correctness_score"] == fact["correctness_score"]
    assert span["attributes"]["lab.evaluator.confidence"] == fact["confidence"]


def test_missing_legacy_score_is_unavailable_not_copied_from_confidence():
    legacy = {
        "fact_index": 0,
        "expected_fact_indices": [0],
        "meaning_match": "equivalent",
        "source_support": "supported",
        "confidence": 95,
        "explanation": "Historical judgment.",
    }
    before = deepcopy(legacy)
    parsed = FactJudgment.model_validate(legacy)
    assert parsed.correctness_score is None
    assert parsed.confidence == 95
    assert legacy == before


@pytest.mark.parametrize("score", [-1, 101, float("nan"), float("inf")])
def test_correctness_score_has_finite_zero_to_hundred_range(score):
    with pytest.raises(ValidationError):
        FactJudgment.model_validate(
            {
                "fact_index": 0,
                "expected_fact_indices": [0],
                "meaning_match": "equivalent",
                "source_support": "supported",
                "correctness_score": score,
                "confidence": 100,
                "explanation": "Test.",
            }
        )


def test_unavailable_evaluation_does_not_use_fifty_as_a_missing_score():
    value = FactJudgment.model_validate(
        {
            "fact_index": 0,
            "expected_fact_indices": [],
            "meaning_match": "not_evaluated",
            "source_support": "not_evaluated",
            "correctness_score": 50,
            "confidence": 80,
            "explanation": "No reference or verified evidence.",
        }
    )
    assert value.correctness_score is None
    assert value.confidence is None


def test_insufficient_support_can_be_a_confident_negative_assessment():
    value = FactJudgment.model_validate(
        {
            "fact_index": 0,
            "expected_fact_indices": [0],
            "meaning_match": "equivalent",
            "source_support": "insufficient_evidence",
            "correctness_score": 10,
            "confidence": 100,
            "explanation": "Quote mentions the amount but does not establish approval.",
        }
    )
    assert value.correctness_score == 10 and value.confidence == 100


def test_answer_unavailable_and_legacy_nullable_scores():
    from traceability_lab.semantic_evaluation import AnswerJudgment

    old = {
        "meaning_match": "equivalent",
        "evidence_support": "supported",
        "confidence": 95,
        "explanation": "Historical.",
    }
    assert AnswerJudgment.model_validate(old).correctness_score is None
    unavailable = AnswerJudgment.model_validate(
        {**old, "meaning_match": "not_evaluated", "correctness_score": 50}
    )
    assert unavailable.correctness_score is None and unavailable.confidence is None


def test_unassessed_support_does_not_erase_assessed_semantic_correctness():
    value = FactJudgment.model_validate(
        {
            "fact_index": 0,
            "expected_fact_indices": [0],
            "meaning_match": "equivalent",
            "source_support": "not_evaluated",
            "correctness_score": 100,
            "confidence": 95,
            "explanation": "Meaning assessed independently.",
        }
    )
    assert value.correctness_score == 100 and value.confidence == 95
