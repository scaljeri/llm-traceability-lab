"""Diagnostics explain strict failures without normalizing or accepting quotes."""

import pytest

from traceability_lab.fact_answers import quote_errors

SOURCE = "Project Orion\n\nProject manager: Anna de Vries.\n\nThe requested budget is EUR 2.4 million.\n\nNo budget has been approved."


@pytest.mark.parametrize(
    ("quote", "code"),
    [
        (SOURCE, None),
        (SOURCE.replace("\n\n", "\n"), "whitespace_mismatch"),
        (SOURCE.replace("\n", "\r\n"), "whitespace_mismatch"),
        ("Project manager: Anna de Vries. ... No budget has been approved.", "non_contiguous_quote"),
        ("The requested budget is EUR 9.9 million.", "text_mismatch"),
        ("Completely unrelated invented statement.", "quote_not_found"),
    ],
)
def test_strict_quotes_and_diagnostics(quote, code):
    answer = {"facts": [{"sources": [{"document_id": "orion", "version": "1", "quote": quote}]}]}
    context = [{"document_id": "orion", "version": "1", "text": SOURCE}]
    errors = quote_errors(answer, context, required=True)
    if code is None:
        assert errors == []
        return
    assert errors[0]["code"] == code
    assert errors[0]["document_id"] == "orion"
    assert errors[0]["version"] == "1"
    assert errors[0]["source"]["quote"] == quote
    if code != "quote_not_found":
        assert "first_difference" in errors[0]
        assert errors[0]["source_excerpt"]
    assert context[0]["text"] == SOURCE


def test_ambiguous_candidate_passages_use_generic_diagnostic():
    answer = {
        "facts": [
            {
                "sources": [
                    {
                        "document_id": "orion",
                        "version": "1",
                        "quote": "The requested budget is EUR 9.9 million.",
                    }
                ]
            }
        ]
    }
    context = [
        {"document_id": "orion", "version": "1", "text": "The requested budget is EUR 2.4 million."},
        {"document_id": "orion", "version": "1", "text": "The requested budget is EUR 3.5 million."},
    ]
    error = quote_errors(answer, context, required=True)[0]
    assert error["code"] == "quote_validation_failed"
    assert "source_excerpt" not in error
