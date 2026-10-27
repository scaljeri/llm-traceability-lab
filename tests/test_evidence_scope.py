"""Evidence routing and independent outcomes; mock judgments are not model accuracy tests."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from traceability_lab.semantic_evaluation import ANSWER_EVALUATOR_POLICY, FACT_EVALUATOR_POLICY


@pytest.mark.parametrize(
    "claim,cited,meaning,support",
    [
        ("Revision BR-07 is EUR 2.85 million.", ["orion-finance-revision"], "equivalent", "supported"),
        (
            "BR-07 was approved at EUR 2.85 million.",
            ["orion-finance-revision", "orion-steerco"],
            "equivalent",
            "supported",
        ),
        (
            "BR-07 was approved at EUR 2.85 million.",
            ["orion-finance-revision"],
            "equivalent",
            "insufficient_evidence",
        ),
        (
            "BR-07 was approved on 18 September 2026.",
            ["orion-finance-revision"],
            "equivalent",
            "insufficient_evidence",
        ),
        ("BR-06 is the approved baseline.", ["orion-finance-revision"], "not_equivalent", "contradicted"),
        ("BR-07 was approved.", ["orion-steerco"], "partial", "supported"),
        (
            "On 18 September 2026 the committee authorized baseline BR-07.",
            ["orion-steerco"],
            "equivalent",
            "supported",
        ),
    ],
)
def test_experiment07_evidence_routing_and_separate_answer_outcome(
    client, monkeypatch, claim, cited, meaning, support
):
    original = json.loads(Path("experiments/07-evidence-hidden-across-multiple-sources.json").read_text())[
        "experiment"
    ]
    definition = deepcopy(original)
    definition["semantic_evaluator"] = {"llm_profile_id": "test-reviewer", "confidence_threshold": None}
    docs = {doc["document_id"]: doc for doc in definition["documents"]}
    sources = [{"document_id": key, "version": "1", "quote": docs[key]["content"]} for key in cited]
    # A single expected claim isolates source completeness from reference coverage.
    definition["semantic_reference"]["facts"] = [
        {
            "fact": claim + " Including the approval date."
            if meaning == "partial"
            else claim + " Reference wording.",
            "sources": sources,
        }
    ]
    generated = {
        "facts": [{"fact": claim, "sources": sources}],
        "answer": "EUR 2.85 million, approved on 18 September 2026.",
    }
    calls = []

    def response(value):
        return {"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]}

    monkeypatch.setattr("traceability_lab.pipeline.generate", lambda *args: response(generated))

    def reviewer(settings, body, chunks):
        payload = json.loads(body["messages"][1]["content"])
        calls.append(body)
        if "generated_fact" in payload:
            assert payload["generated_fact"] == generated["facts"][0]
            assert {item["document_id"] for item in payload["supplied_sources"]} == set(cited)
            assert all(
                item["text"] in docs[item["document_id"]]["content"] for item in payload["supplied_sources"]
            )
            return response(
                {
                    "fact_index": 0,
                    "expected_fact_indices": [0] if meaning == "equivalent" else [],
                    "meaning_match": meaning,
                    "source_support": support,
                    "correctness_score": 100 if meaning == "equivalent" else 50,
                    "confidence": 100,
                    "explanation": "Controlled judgment: cited support and semantic meaning are independent.",
                }
            )
        assert set(payload) == {"question", "expected_answer", "generated_answer", "documents"}
        assert len(payload["documents"]) == len(docs)
        assert all(doc["text"] == docs[doc["document_id"]]["content"] for doc in payload["documents"])
        return response(
            {
                "answer": {
                    "meaning_match": "equivalent",
                    "evidence_support": "supported",
                    "correctness_score": 100,
                    "confidence": 100,
                    "explanation": "The complete collection establishes amount and date.",
                },
                "unsupported_answer": False,
            }
        )

    monkeypatch.setattr("traceability_lab.semantic_evaluation.generate", reviewer)
    response_ = client.post(
        "/api/experiment-definitions/run",
        json={
            "definition": definition,
            "execution": {"repetitions": 1, "llm_profile_id": "test-generator"},
        },
    )
    assert response_.status_code == 201, response_.text
    batch = client.get(f"/api/batches/{response_.json()['batch_id']}").json()
    criteria = {item["id"]: item for item in batch["evaluations"][0]["result_criteria"]}
    assert criteria["answer"]["result"] == "pass"
    assert criteria["facts"]["result"] == (
        "pass" if meaning == "equivalent" and support == "supported" else "fail"
    ), batch["evaluations"][0]["automatic"]["semantic_evaluation"]
    assert len(calls) == 2
    assert (
        json.loads(Path("experiments/07-evidence-hidden-across-multiple-sources.json").read_text())[
            "experiment"
        ]
        == original
    )


def test_generic_policies_define_complete_claim_and_evidence_boundaries():
    assert "candidate_expected_facts are comparison targets, never evidence" in FACT_EVALUATOR_POLICY
    assert "all necessary evidence must be cited" in FACT_EVALUATOR_POLICY
    assert "use insufficient_evidence, not supported" in FACT_EVALUATOR_POLICY
    assert "use partial" in FACT_EVALUATOR_POLICY.lower()
    assert "cross-document claims" in ANSWER_EVALUATOR_POLICY
    assert "use partial" in ANSWER_EVALUATOR_POLICY
    for policy in (FACT_EVALUATOR_POLICY, ANSWER_EVALUATOR_POLICY):
        assert all(term not in policy.lower() for term in ("orion", "budget", "finance", "steering"))
