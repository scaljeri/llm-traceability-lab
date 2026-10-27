"""Opt-in reviewer behavior regressions using public fictional experiments."""

import json
import os
from pathlib import Path

import httpx
import pytest

from traceability_lab.config import service_url
from traceability_lab.semantic_evaluation import ANSWER_EVALUATOR_POLICY, ANSWER_OUTPUT_FORMAT, AnswerReview


def review_cases():
    definitions = {
        number: json.loads(next(Path("experiments").glob(f"{number:02}-*.json")).read_text())["experiment"]
        for number in range(1, 5)
    }
    examples = [
        ("manager_paraphrase", 1, "Project Orion is managed by Anna de Vries.", True),
        ("missing_approver", 2, "The supplied information does not identify who approved the budget.", True),
        (
            "conflicting_evidence",
            3,
            "The sources report different approved budgets: EUR 2.4 million and EUR 2.6 million. The evidence does not determine an unambiguous approved amount.",
            True,
        ),
        ("document_limited", 4, "The supplied document does not establish an approved budget.", True),
        ("categorical_nonexistence", 4, "There is no approved budget.", False),
        ("proposal_as_approved", 4, "The approved budget is EUR 2.6 million.", False),
        ("uncertain_as_certain", 4, "The budget proposal will definitely be approved.", False),
        ("missing_as_nonexistence", 2, "No person approved the budget of Project Orion.", False),
    ]
    return [
        (
            name,
            {
                "question": definitions[number]["question"],
                "expected_answer": definitions[number]["semantic_reference"]["answer"],
                "generated_answer": answer,
                "documents": [
                    {
                        "document_id": doc["document_id"],
                        "version": str(doc["version"]),
                        "text": doc["content"],
                    }
                    for doc in definitions[number]["documents"]
                    if doc.get("enabled", True)
                ],
            },
            expected,
        )
        for name, number, answer, expected in examples
    ]


def request_for_case(model, payload, *, structured=False):
    policy = ANSWER_EVALUATOR_POLICY
    if not structured:
        policy += "\n" + ANSWER_OUTPUT_FORMAT
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": policy},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
        ],
        "temperature": 0,
        "max_tokens": 4096,
        "stream": False,
    }
    if structured:
        schema = AnswerReview.model_json_schema()
        schema["$defs"]["AnswerJudgment"]["required"] = list(schema["$defs"]["AnswerJudgment"]["properties"])
        body["response_format"] = {"type": "json_object", "schema": schema}
    return body


def accepted(judgment):
    return (
        judgment["answer"]["meaning_match"] == "equivalent"
        and judgment["answer"]["evidence_support"] == "supported"
        and judgment["unsupported_answer"] is False
    )


def test_experiment04_reference_is_an_answer_not_evaluation_instructions():
    definitions = dict((name, payload) for name, payload, _ in review_cases())
    assert definitions["document_limited"]["expected_answer"] == (
        "The approved budget cannot be determined from the supplied document. "
        "EUR 2.6 million is the proposed budget, and approval is still pending."
    )
    assert set(definitions["document_limited"]) == {
        "question",
        "expected_answer",
        "generated_answer",
        "documents",
    }


@pytest.mark.model
@pytest.mark.skipif(
    os.environ.get("LAB_RUN_ANSWER_REVIEW_TESTS") != "1", reason="Explicit live reviewer opt-in required."
)
@pytest.mark.parametrize(
    "name,payload,expected", review_cases(), ids=lambda value: value if isinstance(value, str) else None
)
def test_live_generic_answer_scope(name, payload, expected):
    url = os.environ.get("LAB_ANSWER_REVIEW_URL")
    model = os.environ.get("LAB_ANSWER_REVIEW_MODEL")
    assert url and model, "Configure LAB_ANSWER_REVIEW_URL and LAB_ANSWER_REVIEW_MODEL explicitly."
    body = request_for_case(model, payload, structured=os.environ.get("LAB_ANSWER_REVIEW_SCHEMA") == "1")
    with httpx.Client(timeout=180, trust_env=False, follow_redirects=False) as client:
        response = client.post(service_url(url) + "/chat/completions", json=body)
        response.raise_for_status()
    judgment = AnswerReview.model_validate_json(
        response.json()["choices"][0]["message"]["content"]
    ).model_dump()
    assert accepted(judgment) == expected, (name, judgment)


def test_same_correct_facts_cannot_change_answer_review_input():
    from traceability_lab.semantic_evaluation import answer_review_payload

    definition = json.loads(Path("experiments/04-perfect-citation.json").read_text())["experiment"]
    run = {
        "question": definition["question"],
        "snapshot": {"evaluation_reference": {"definition": definition}},
        "context": [
            {
                "document_id": doc["document_id"],
                "version": doc["version"],
                "text": doc["content"],
                "start": 0,
                "end": len(doc["content"]),
            }
            for doc in definition["documents"]
        ],
    }
    facts = definition["semantic_reference"]["facts"]
    supported = answer_review_payload(
        run,
        {"facts": facts, "answer": "The supplied document does not establish an approved budget."},
        definition["semantic_reference"],
    )
    unsupported = answer_review_payload(
        run, {"facts": facts, "answer": "There is no approved budget."}, definition["semantic_reference"]
    )
    assert supported.pop("generated_answer") != unsupported.pop("generated_answer")
    assert supported == unsupported
    assert set(supported) == {"question", "expected_answer", "documents"}
