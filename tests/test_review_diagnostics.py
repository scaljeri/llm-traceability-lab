"""Keep deterministic citation errors visible alongside semantic review errors."""

import pytest

from traceability_lab.fact_answers import FACT_ANSWER_JSON, fact_result_criteria


@pytest.mark.parametrize("review_error", [None, "Reviewer unavailable"])
def test_citation_failure_is_always_in_the_reason(review_error):
    run = {
        "backend": "local",
        "status": "completed",
        "answer": '{"facts":[],"answer":null}',
        "snapshot": {
            "evaluation_reference": {
                "definition": {"answer_json": FACT_ANSWER_JSON, "semantic_reference": None}
            }
        },
    }
    automatic = {
        "criteria": [{"id": "answer_json", "result": "pass"}, {"id": "source_ids", "result": "fail"}],
        "semantic_evaluation": {"fact_evaluation_errors": [review_error] if review_error else []},
    }
    facts = next(item for item in fact_result_criteria(run, automatic) if item["id"] == "facts")
    assert facts["result"] == "fail"
    assert facts["reason"].startswith("Deterministic citation validation failed.")
    if review_error:
        assert review_error in facts["reason"]
