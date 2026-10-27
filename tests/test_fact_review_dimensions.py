"""Public experiment 07 cases: deterministic routing versus opt-in model judgments."""

import json
import os
from copy import deepcopy
from pathlib import Path

import httpx
import pytest

from traceability_lab.config import service_url
from traceability_lab.fact_answers import quote_errors
from traceability_lab.semantic_evaluation import (
    FACT_EVALUATOR_POLICY,
    FactJudgment,
    candidate_facts,
)


def fact_review_cases():
    definition = json.loads(Path("experiments/07-evidence-hidden-across-multiple-sources.json").read_text())[
        "experiment"
    ]
    reference = definition["semantic_reference"]
    docs = {doc["document_id"]: doc for doc in definition["documents"]}
    approval = reference["facts"][0]["fact"]
    amount = reference["facts"][1]["fact"]
    combined = (
        "The steering committee approved Project Orion baseline BR-07 at EUR 2.85 million "
        "on 18 September 2026."
    )
    specifications = [
        ("equivalent", approval, ["orion-steerco"], "equivalent", "supported", 100),
        (
            "wrong_cited_document",
            approval,
            ["orion-finance-revision"],
            "equivalent",
            "insufficient_evidence",
            100,
        ),
        (
            "explicit_two_document_link",
            combined,
            ["orion-steerco", "orion-finance-revision"],
            "equivalent",
            "supported",
            100,
        ),
        (
            "amount_does_not_prove_approval",
            combined,
            ["orion-finance-revision"],
            "equivalent",
            "insufficient_evidence",
            100,
        ),
        (
            "missing_essential_date",
            "The steering committee approved Project Orion baseline BR-07.",
            ["orion-steerco"],
            "partial",
            "supported",
            None,
        ),
        (
            "supported_unexpected",
            "Project Orion targets deployment in Q2 2027.",
            ["orion-charter"],
            "not_equivalent",
            "supported",
            None,
        ),
        (
            "invalid_quote_underlying_support",
            amount,
            ["orion-finance-revision"],
            "equivalent",
            "supported",
            100,
        ),
        (
            "confidently_incorrect",
            "The steering committee rejected Project Orion baseline BR-07 on 18 September 2026.",
            ["orion-steerco"],
            "not_equivalent",
            "contradicted",
            0,
        ),
    ]
    cases = []
    for name, claim, ids, meaning, support, score in specifications:
        sources = [{"document_id": key, "version": "1", "quote": docs[key]["content"]} for key in ids]
        if name == "invalid_quote_underlying_support":
            sources[0]["quote"] = sources[0]["quote"].replace("\n\n", "\n")
        fact = {"fact": claim, "sources": sources}
        context = [
            {
                "document_id": key,
                "version": "1",
                "chunk_id": key,
                "start": 0,
                "end": len(docs[key]["content"]),
                "text": docs[key]["content"],
            }
            for key in ids
        ]
        payload = {
            "research_question": definition["question"],
            "fact_index": 0,
            "generated_fact": fact,
            "candidate_expected_facts": candidate_facts(fact, reference, context),
            "reference_available": True,
            "source_mapping_verified": True,
            "supplied_sources": context,
            "literal_quote_errors": quote_errors({"facts": [fact]}, context, required=True),
            "output_schema": FactJudgment.model_json_schema(),
        }
        cases.append((name, payload, {"meaning_match": meaning, "source_support": support, "score": score}))
    return cases


def fact_request(model, payload):
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": FACT_EVALUATOR_POLICY},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "temperature": 0,
        "max_tokens": 4096,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def assert_fact_judgment(judgment, expected):
    assert judgment["meaning_match"] == expected["meaning_match"], judgment
    assert judgment["source_support"] == expected["source_support"], judgment
    if expected["score"] is not None:
        assert judgment["correctness_score"] == expected["score"], judgment
    else:
        assert 0 <= judgment["correctness_score"] < 100, judgment
    if expected["meaning_match"] in {"equivalent", "partial"}:
        assert judgment["expected_fact_indices"], judgment
    else:
        assert judgment["expected_fact_indices"] == [], judgment


def test_all_candidates_remain_available_but_only_cited_sources_are_evidence():
    for name, payload, _ in fact_review_cases():
        assert [item["expected_fact_index"] for item in payload["candidate_expected_facts"]] == [0, 1]
        assert {source["document_id"] for source in payload["supplied_sources"]} == {
            source["document_id"] for source in payload["generated_fact"]["sources"]
        }
        assert bool(payload["literal_quote_errors"]) == (name == "invalid_quote_underlying_support")


@pytest.mark.parametrize("name,payload,expected", fact_review_cases())
def test_independent_judgments_survive_schema_validation(name, payload, expected):
    raw = {
        "fact_index": 0,
        "expected_fact_indices": [0] if expected["meaning_match"] != "not_equivalent" else [],
        "meaning_match": expected["meaning_match"],
        "source_support": expected["source_support"],
        "correctness_score": expected["score"] if expected["score"] is not None else 60,
        "confidence": 97,
        "explanation": "Controlled independent dimension assessment.",
    }
    original = deepcopy(raw)
    assert FactJudgment.model_validate(raw).model_dump() == original


@pytest.mark.model
@pytest.mark.skipif(
    os.environ.get("LAB_RUN_FACT_REVIEW_TESTS") != "1", reason="Explicit live reviewer opt-in required."
)
@pytest.mark.parametrize("name,payload,expected", fact_review_cases())
def test_live_fact_dimensions(name, payload, expected):
    url = os.environ.get("LAB_FACT_REVIEW_URL")
    model = os.environ.get("LAB_FACT_REVIEW_MODEL")
    assert url and model, "Configure LAB_FACT_REVIEW_URL and LAB_FACT_REVIEW_MODEL explicitly."
    with httpx.Client(timeout=180, trust_env=False, follow_redirects=False) as client:
        response = client.post(service_url(url) + "/chat/completions", json=fact_request(model, payload))
        response.raise_for_status()
    judgment = FactJudgment.model_validate_json(
        response.json()["choices"][0]["message"]["content"]
    ).model_dump()
    assert_fact_judgment(judgment, expected)
    if name == "confidently_incorrect":
        assert judgment["confidence"] > judgment["correctness_score"]


@pytest.mark.parametrize("name,payload,expected", fact_review_cases())
def test_pipeline_preserves_dimensions_and_deterministic_quote_failures(
    client, monkeypatch, name, payload, expected
):
    definition = json.loads(Path("experiments/07-evidence-hidden-across-multiple-sources.json").read_text())[
        "experiment"
    ]
    original_reference = deepcopy(definition["semantic_reference"])
    definition["semantic_evaluator"] = {"llm_profile_id": "test-reviewer", "confidence_threshold": None}
    generated = {
        "facts": [
            payload["generated_fact"],
            *[{"fact": item["fact"], "sources": item["sources"]} for item in original_reference["facts"]],
        ],
        "answer": "EUR 2.85 million, approved on 18 September 2026.",
    }
    calls = []

    from traceability_lab.pipeline import read_run

    def read_without_annotation_queries(db, run_id, **kwargs):
        rows = db.rows

        def checked_rows(sql, *args, **options):
            assert not any(f"from {table}" in sql.lower() for table in ("claims", "assessments")), sql
            return rows(sql, *args, **options)

        with monkeypatch.context() as patch:
            patch.setattr(db, "rows", checked_rows)
            return read_run(db, run_id, **kwargs)

    monkeypatch.setattr("traceability_lab.semantic_evaluation.read_run", read_without_annotation_queries)

    def envelope(value):
        return {"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]}

    monkeypatch.setattr("traceability_lab.pipeline.generate", lambda *args: envelope(generated))

    def reviewer(settings, body, chunks):
        data = json.loads(body["messages"][1]["content"])
        calls.append(deepcopy(body))
        if "generated_fact" not in data:
            return envelope(
                {
                    "answer": {
                        "meaning_match": "equivalent",
                        "evidence_support": "supported",
                        "correctness_score": 100,
                        "confidence": 98,
                        "explanation": "Controlled answer judgment independent of fact citations.",
                    },
                    "unsupported_answer": False,
                }
            )
        assert len(data["candidate_expected_facts"]) == len(original_reference["facts"])
        cited = {(item["document_id"], item["version"]) for item in data["generated_fact"]["sources"]}
        assert {(item["document_id"], item["version"]) for item in data["supplied_sources"]} == cited
        for document_id, version in cited:
            actual_chunks = sorted(
                [
                    item
                    for item in data["supplied_sources"]
                    if item["document_id"] == document_id and item["version"] == version
                ],
                key=lambda item: item["start"],
            )
            document = next(
                doc
                for doc in definition["documents"]
                if doc["document_id"] == document_id and doc["version"] == version
            )
            assert "".join(item["text"] for item in actual_chunks) == document["content"]
        index = data["fact_index"]
        if index:
            meaning, support, score, matches = "equivalent", "supported", 100, [index - 1]
        else:
            meaning, support = expected["meaning_match"], expected["source_support"]
            score = expected["score"] if expected["score"] is not None else 60
            matches = (
                []
                if meaning == "not_equivalent"
                else [1]
                if name == "invalid_quote_underlying_support"
                else [0, 1]
                if name in {"explicit_two_document_link", "amount_does_not_prove_approval"}
                else [0]
            )
        return envelope(
            {
                "fact_index": index,
                "expected_fact_indices": matches,
                "meaning_match": meaning,
                "source_support": support,
                "correctness_score": score,
                "confidence": 97,
                "explanation": "Controlled semantic judgment; citation validation is separate.",
            }
        )

    monkeypatch.setattr("traceability_lab.semantic_evaluation.generate", reviewer)
    response = client.post(
        "/api/experiment-definitions/run",
        json={"definition": definition, "execution": {"repetitions": 1, "llm_profile_id": "test-generator"}},
    )
    assert response.status_code == 201, response.text
    batch = client.get(f"/api/batches/{response.json()['batch_id']}").json()
    evaluation = batch["evaluations"][0]
    observation = evaluation["automatic"]["semantic_evaluation"]
    judgment = observation["judgments"]["facts"][0]
    assert_fact_judgment(judgment, expected)
    assert judgment["confidence"] == 97
    assert observation["calls"][0]["request_body"] == calls[0]
    criteria = {item["id"]: item for item in evaluation["result_criteria"]}
    assert criteria["answer"]["result"] == "pass"
    fails = (
        expected["meaning_match"] != "equivalent"
        or expected["source_support"] != "supported"
        or name == "invalid_quote_underlying_support"
    )
    assert criteria["facts"]["result"] == ("fail" if fails else "pass")
    assert definition["semantic_reference"] == original_reference
    assert (
        json.loads(Path("experiments/07-evidence-hidden-across-multiple-sources.json").read_text())[
            "experiment"
        ]["semantic_reference"]
        == original_reference
    )
