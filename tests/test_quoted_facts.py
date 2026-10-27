"""Quoted source contracts and source-filtered, individually recorded model judgments."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from traceability_lab.db import json_text
from traceability_lab.fact_answers import FACT_ANSWER_JSON, quote_errors
from traceability_lab.models import ExperimentDefinition
from traceability_lab.semantic_evaluation import candidate_facts


def quoted_experiment():
    data = json.loads(Path("experiments/01-complete-experiment.json").read_text())["experiment"]
    data["semantic_evaluator"] = {"llm_profile_id": "test-reviewer", "confidence_threshold": None}
    data["documents"] += [
        {"document_id": "atlas", "version": "1", "content": "Atlas budget: 42 euros."},
        {"document_id": "orion", "version": "2", "content": "Project manager: Someone else."},
    ]
    data["semantic_reference"]["facts"] += [
        {
            "fact": "Atlas budget is 42 euros.",
            "sources": [{"document_id": "atlas", "version": "1", "quote": "Atlas budget: 42 euros."}],
        },
        {
            "fact": "Someone else manages Orion.",
            "sources": [{"document_id": "orion", "version": "2", "quote": "Project manager: Someone else."}],
            "optional": True,
        },
    ]
    return data


def quoted_answer():
    return {
        "facts": [
            {
                "fact": "Atlas has a budget of 42 euros.",
                "sources": [{"document_id": "atlas", "version": "1", "quote": "Atlas budget: 42 euros."}],
            },
            {
                "fact": "Anna manages Orion.",
                "sources": [
                    {
                        "document_id": "orion",
                        "version": "1",
                        "quote": "Project Orion\n\nProject manager: Anna de Vries.",
                    }
                ],
            },
        ],
        "answer": "Anna de Vries manages Project Orion.",
    }


def configure_calls(monkeypatch, *, failure=None, bad_quote=False):
    calls = []
    answer = quoted_answer()
    if bad_quote:
        answer["facts"][1]["sources"][0]["quote"] = "A fabricated passage."

    def generator(settings, body, chunks):
        text = json_text(body)
        assert FACT_ANSWER_JSON["facts"][0]["fact"] in text
        assert "candidate_expected_facts" not in text
        assert "Someone else manages Orion." not in text
        return {"choices": [{"message": {"content": json.dumps(answer)}, "finish_reason": "stop"}]}

    def evaluator(settings, body, chunks):
        payload = json.loads(body["messages"][1]["content"])
        calls.append(deepcopy(body))
        if "generated_fact" in payload:
            index = payload["fact_index"]
            assert [item["expected_fact_index"] for item in payload["candidate_expected_facts"]] == [
                0,
                1,
                2,
                3,
            ]
            assert {item["document_id"] for item in payload["supplied_sources"]} == (
                {"atlas"} if index == 0 else {"orion"}
            )
            assert all(item["version"] == "1" for item in payload["supplied_sources"])
            assert bool(payload["literal_quote_errors"]) == (bad_quote and index == 1)
            assert all(error["fact_index"] == index for error in payload["literal_quote_errors"])
            if failure == "http":
                raise OSError("Evaluator unavailable")
            if failure == "invalid":
                return {"choices": [{"message": {"content": "not json"}, "finish_reason": "stop"}]}
            value = {
                "fact_index": index,
                "expected_fact_indices": [4 if failure == "wrong_candidate" else 2 if index == 0 else 0],
                "meaning_match": "equivalent",
                "source_support": "supported",
                "explanation": "Recorded per-fact judgment.",
            }
        else:
            assert set(payload) == {"question", "expected_answer", "generated_answer", "documents"}
            assert payload["generated_answer"] == answer["answer"]
            value = {
                "expected_facts": [
                    {
                        "expected_fact_index": i,
                        "status": "covered" if i in (0, 2) else "missing",
                        "explanation": "Recorded coverage.",
                    }
                    for i in range(4)
                ],
                "answer": {
                    "meaning_match": "equivalent",
                    "evidence_support": "supported",
                    "explanation": "Supported by generated facts.",
                },
                "unsupported_fact_indices": [],
                "conflicting_fact_indices": [],
                "unsupported_answer": False,
            }
            value = {
                "answer": value["answer"],
                "unsupported_answer": value["unsupported_answer"],
            }
            if failure == "fact_override":
                value["conflicting_fact_indices"] = [0]
        return {
            "choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 30},
        }

    monkeypatch.setattr("traceability_lab.pipeline.generate", generator)
    monkeypatch.setattr("traceability_lab.semantic_evaluation.generate", evaluator)
    return calls


def execute(client):
    response = client.post(
        "/api/experiment-definitions/run",
        json={
            "definition": quoted_experiment(),
            "execution": {"repetitions": 1, "llm_profile_id": "test-generator"},
        },
    )
    assert response.status_code == 201, response.text
    summary = client.get(f"/api/batches/{response.json()['batch_id']}").json()
    return summary, summary["evaluations"][0]["automatic"]["semantic_evaluation"]


def test_each_fact_is_source_filtered_and_calls_are_persisted(client, monkeypatch):
    calls = configure_calls(monkeypatch)
    summary, observation = execute(client)
    assert observation["status"] == "completed", observation
    assert observation["prompt_version"] == "facts-semantic-v15"
    assert len(calls) == 3
    assert [item["phase"] for item in observation["calls"]] == ["fact", "fact", "answer"]
    assert [item["fact_index"] for item in observation["calls"]] == [0, 1, None]
    for entry, body in zip(observation["calls"], calls, strict=True):
        assert entry["status"] == "completed"
        assert entry["request_body"] == body
        assert entry["raw_answer"] == entry["response"]["choices"][0]["message"]["content"]
        assert entry["span_id"] and entry["started_at"] and entry["finished_at"]
        assert entry["duration_ms"] >= 0
    assert all(item["pass"] == 1 for item in summary["property_counts"])
    assert (
        client.get(f"/api/runs/{summary['runs'][0]['id']}/evaluation").json()["automatic"][
            "semantic_evaluation"
        ]
        == observation
    )
    assert summary["evaluations"][0]["reference"]["definition"]["answer_json"] == FACT_ANSWER_JSON


@pytest.mark.parametrize("failure", ["wrong_candidate", "invalid", "http", "fact_override"])
def test_evaluator_failures_keep_input_and_raw_output(client, monkeypatch, failure):
    configure_calls(monkeypatch, failure=failure)
    summary, observation = execute(client)
    assert summary["runs"][0]["status"] == "completed"
    assert observation["status"] == "failed", observation
    failed = [call for call in observation["calls"] if call["status"] == "failed"]
    assert failed
    assert all(call["request_body"] and call["error"] for call in failed)
    if failure != "http":
        assert all(call["raw_answer"] for call in failed)
    assert summary["property_counts"][1]["pass" if failure == "fact_override" else "evaluation_error"] == 1
    assert summary["property_counts"][2]["evaluation_error" if failure == "fact_override" else "pass"] == 1


def test_fabricated_quote_fails_even_when_evaluator_reports_supported(client, monkeypatch):
    configure_calls(monkeypatch, bad_quote=True)
    summary, observation = execute(client)
    assert observation["status"] == "completed"
    assert summary["property_counts"][1]["fail"] == 1
    assert "Quote" in summary["evaluations"][0]["result_criteria"][1]["reason"]


def test_quotes_span_only_adjacent_supplied_chunks():
    answer = {"facts": [{"sources": [{"document_id": "d", "version": "1", "quote": "bc"}]}]}
    context = [
        {"document_id": "d", "version": "1", "text": "ab", "start": 0, "end": 2},
        {"document_id": "d", "version": "1", "text": "cd", "start": 2, "end": 4},
    ]
    assert quote_errors(answer, context, required=True) == []
    context[1]["start"] = 9
    assert quote_errors(answer, context, required=True)
    answer["facts"][0]["sources"][0].pop("quote")
    assert quote_errors(answer, context, required=True)
    assert quote_errors(answer, context) == []


def test_candidate_selection_keeps_semantic_targets_even_when_citations_differ():
    reference = {
        "facts": [
            {"fact": "Different version", "sources": [{"document_id": "a", "version": "2"}]},
            {
                "fact": "Same source",
                "sources": [{"document_id": "a", "version": "1", "quote": "different quote"}],
            },
            {"fact": "Other source", "sources": [{"document_id": "b", "version": "1"}]},
        ]
    }
    assert [
        item["expected_fact_index"]
        for item in candidate_facts(
            {"sources": [{"document_id": "a", "version": "1", "quote": "actual quote"}]}, reference, []
        )
    ] == [0, 1, 2]


def test_import_export_uses_fixed_descriptive_output_and_preserves_expected_quotes(client):
    definition = quoted_experiment()
    parsed = ExperimentDefinition.model_validate(definition).model_dump()
    assert parsed["answer_json"] == FACT_ANSWER_JSON
    response = client.post("/api/experiment-definitions", json=definition)
    assert response.status_code == 201, response.text
    exported = client.get(f"/api/experiment-definitions/{response.json()['id']}/export").json()
    assert "answer_json" not in exported["experiment"]
    imported = client.post("/api/experiment-definitions/import", json=exported)
    assert imported.status_code == 200, imported.text
    assert imported.json()["definition"]["semantic_reference"] == parsed["semantic_reference"]


def test_missing_quote_fails_new_contract_without_calling_evaluator(client, monkeypatch):
    answer = quoted_answer()
    answer["facts"][0]["sources"][0].pop("quote")
    monkeypatch.setattr(
        "traceability_lab.pipeline.generate",
        lambda *args: {"choices": [{"message": {"content": json.dumps(answer)}, "finish_reason": "stop"}]},
    )
    monkeypatch.setattr(
        "traceability_lab.semantic_evaluation.generate",
        lambda *args: pytest.fail("Malformed output must not be evaluated"),
    )
    summary, observation = execute(client)
    assert observation["status"] == "not_evaluated"
    assert summary["property_counts"][0]["fail"] == 1
    assert summary["property_counts"][1]["fail"] == 1


def test_quote_output_contract_preview_matches_actual_system_prompt(client, monkeypatch):
    configure_calls(monkeypatch)
    definition = quoted_experiment()
    preview = client.post(
        "/api/context-preview",
        json={
            "question": definition["question"],
            "documents": definition["documents"],
            "context_mode": "explicit",
            "context_ids": [],
            "system_prompt": definition.get("system_prompt"),
            "prompt_instructions": definition.get("context", ""),
            "answer_json": FACT_ANSWER_JSON,
            "output_mode": "facts_and_answer",
        },
    )
    assert preview.status_code == 200, preview.text
    assert '"quote"' in preview.json()["system_prompt"]
    summary, _ = execute(client)
    run = client.get(f"/api/runs/{summary['runs'][0]['id']}").json()
    assert run["messages"][0]["content"] == preview.json()["system_prompt"]
    assert run["messages"][0]["content"].count("OUTPUT CONTRACT") == 1
    assert run["messages"][-1]["content"] == preview.json()["text"]


def test_fixed_output_contract_replaces_repeated_slots_without_reference_leakage():
    from traceability_lab.llm import build_messages

    definition = quoted_experiment()
    definition["answer_json"] = {"obsolete": "Ignored old template"}
    parsed = ExperimentDefinition.model_validate(definition)
    assert parsed.answer_json == FACT_ANSWER_JSON
    assert (
        "answer_json"
        not in json.loads(Path("experiments/01-complete-experiment.json").read_text())["experiment"]
    )
    slots = "{{output_json}}\n{{output_json}}"
    messages = build_messages(
        "Who manages Orion?",
        [],
        output_mode="facts_and_answer",
        prompt_instructions=slots,
        answer_json={"obsolete": "Ignored old template"},
    )
    text = messages[1]["content"]
    for slot in ("{{output_json}}",):
        assert slot not in text
    assert text.count(FACT_ANSWER_JSON["facts"][0]["sources"][0]["quote"]) == 2
    assert "obsolete" not in json.dumps(messages)
    assert "Anna de Vries" not in json.dumps(messages)
    # The editor's read-only structure must match the backend's actual contract.
    editor = Path("src/traceability_lab/web/experiment-editor.mjs").read_text()
    template = editor.split("export const factAnswerTemplate = ", 1)[1]
    assert json.JSONDecoder().raw_decode(template)[0] == FACT_ANSWER_JSON


@pytest.mark.parametrize("confidence, passed", [(79, False), (80, True), (100, True), (None, False)])
def test_review_confidence_threshold_is_frozen_and_binary(client, monkeypatch, confidence, passed):
    import traceability_lab.semantic_evaluation as semantic

    configure_calls(monkeypatch)
    original = semantic.generate

    def scored(settings, body, chunks):
        response = original(settings, body, chunks)
        value = json.loads(response["choices"][0]["message"]["content"])

        def score(item):
            if isinstance(item, dict):
                if "explanation" in item and confidence is not None:
                    item["confidence"] = confidence
                for child in list(item.values()):
                    score(child)
            elif isinstance(item, list):
                for child in item:
                    score(child)

        score(value)
        response["choices"][0]["message"]["content"] = json.dumps(value)
        return response

    monkeypatch.setattr(semantic, "generate", scored)
    definition = quoted_experiment()
    definition["semantic_evaluator"]["confidence_threshold"] = 80
    response = client.post(
        "/api/experiment-definitions/run",
        json={
            "definition": definition,
            "execution": {"repetitions": 1, "llm_profile_id": "test-generator"},
        },
    )
    assert response.status_code == 201, response.text
    batch = client.get(f"/api/batches/{response.json()['batch_id']}").json()
    evaluation = batch["evaluations"][0]
    assert evaluation["automatic"]["semantic_evaluation"]["configuration"]["confidence_threshold"] == 80
    for criterion in evaluation["result_criteria"][1:]:
        assert criterion["result"] == ("pass" if passed else "fail")
        assert criterion["confidence_threshold"] == 80
        if not passed:
            assert "confidence" in criterion["reason"]


def test_rejected_format_explains_missing_facts_array_in_new_and_saved_results():
    from traceability_lab.fact_answers import evaluate_fact_answer, fact_result_criteria

    run = {
        "answer": json.dumps({"fact": "A statement", "sources": [], "answer": "Answer"}),
        "backend": "local",
        "status": "completed",
        "context": [],
        "snapshot": {
            "evaluation_reference": {
                "definition": {
                    "answer_json": FACT_ANSWER_JSON,
                    "semantic_reference": {"facts": [], "answer": "Answer"},
                }
            }
        },
    }
    automatic = evaluate_fact_answer(run)
    assert automatic["criteria"][0]["result"] == "fail"
    assert 'Missing "facts" array' in automatic["criteria"][0]["reason"]
    automatic["criteria"][0]["reason"] = (
        "Checks facts array, nonempty statements, document/version source pairs."
    )
    projected = fact_result_criteria(run, automatic)
    assert 'Missing "facts" array' in projected[0]["reason"]
    assert automatic["criteria"][0]["reason"].startswith("Checks facts array")


@pytest.mark.parametrize(
    "raw, reason",
    [
        ("not JSON", "Invalid JSON"),
        ('{"fact":"Statement","sources":[],"answer":"Answer"}', 'Missing "facts" array'),
    ],
)
def test_invalid_answer_stops_before_review_and_preserves_raw_response(client, monkeypatch, raw, reason):
    monkeypatch.setattr(
        "traceability_lab.pipeline.generate",
        lambda *args: {
            "choices": [{"message": {"content": raw}, "finish_reason": "stop"}],
        },
    )
    monkeypatch.setattr(
        "traceability_lab.semantic_evaluation.generate",
        lambda *args: pytest.fail("Invalid answer must never reach reviewer"),
    )
    summary, observation = execute(client)
    assert summary["runs"][0]["answer"] == raw
    assert observation["status"] == "not_evaluated"
    assert reason in observation["reason"]
    criteria = summary["evaluations"][0]["result_criteria"]
    assert criteria[0]["result"] == "fail"
    for criterion in criteria[1:]:
        assert criterion["result"] == "fail"
        assert reason in criterion["reason"]
    for counts in summary["property_counts"]:
        assert counts["fail"] == counts["total"] == 1
        assert counts["not_evaluated"] == counts["requires_review"] == counts["pass"] == 0


@pytest.mark.parametrize("accepted", [True, False])
def test_exact_empty_reference_still_calls_reviewer(client, monkeypatch, accepted):
    raw = '{"facts": [], "answer": null}'
    monkeypatch.setattr(
        "traceability_lab.pipeline.generate",
        lambda *args: {
            "choices": [{"message": {"content": raw}, "finish_reason": "stop"}],
        },
    )
    monkeypatch.setattr(
        "traceability_lab.semantic_evaluation.generate",
        lambda *args: {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "answer": {
                                    "meaning_match": "equivalent" if accepted else "not_equivalent",
                                    "evidence_support": "supported",
                                    "correctness_score": 100 if accepted else 0,
                                    "confidence": 100,
                                    "explanation": "Reviewed null.",
                                },
                                "unsupported_answer": False,
                            }
                        )
                    },
                    "finish_reason": "stop",
                }
            ]
        },
    )
    definition = quoted_experiment()
    definition["semantic_reference"] = {"facts": [], "answer": None}
    definition["semantic_evaluator"]["confidence_threshold"] = 100
    result = client.post(
        "/api/experiment-definitions/run",
        json={"definition": definition, "execution": {"repetitions": 1, "llm_profile_id": "test-generator"}},
    )
    assert result.status_code == 201, result.text
    batch = client.get(f"/api/batches/{result.json()['batch_id']}").json()
    evaluation = batch["evaluations"][0]
    observation = evaluation["automatic"]["semantic_evaluation"]
    assert observation["status"] == "completed"
    assert observation.get("method") != "exact_reference"
    assert len(observation["calls"]) == 1
    assert batch["overview"]["evaluator"]["calls"] == 1
    assert batch["property_counts"][-1]["pass" if accepted else "fail"] == 1
    assert evaluation["result_criteria"][-1]["confidence"] == 100


@pytest.mark.parametrize("include_optional", [False, True])
def test_exact_facts_ignore_order_optional_omissions_and_still_call_reviewers(
    client, monkeypatch, include_optional
):
    definition = quoted_experiment()
    reference = definition["semantic_reference"]
    facts = [
        {"fact": fact["fact"], "sources": fact["sources"]}
        for fact in reversed(reference["facts"])
        if include_optional or not fact.get("optional")
    ]
    raw = json.dumps({"facts": facts, "answer": reference["answer"]})
    monkeypatch.setattr(
        "traceability_lab.pipeline.generate",
        lambda *args: {"choices": [{"message": {"content": raw}, "finish_reason": "stop"}]},
    )
    calls = []

    def reviewer(settings, body, chunks):
        payload = json.loads(body["messages"][1]["content"])
        calls.append(payload)
        judgment = {
            "meaning_match": "equivalent",
            "correctness_score": 100,
            "confidence": 100,
            "explanation": "Reviewed exact match.",
        }
        if "generated_fact" in payload:
            expected_index = next(
                i
                for i, fact in enumerate(reference["facts"])
                if fact["fact"] == payload["generated_fact"]["fact"]
            )
            value = {
                **judgment,
                "fact_index": payload["fact_index"],
                "expected_fact_indices": [expected_index],
                "source_support": "supported",
            }
        else:
            value = {"answer": {**judgment, "evidence_support": "supported"}, "unsupported_answer": False}
        return {"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]}

    monkeypatch.setattr("traceability_lab.semantic_evaluation.generate", reviewer)
    result = client.post(
        "/api/experiment-definitions/run",
        json={"definition": definition, "execution": {"repetitions": 1, "llm_profile_id": "test-generator"}},
    )
    assert result.status_code == 201, result.text
    batch = client.get(f"/api/batches/{result.json()['batch_id']}").json()
    assert all(count["pass"] == 1 for count in batch["property_counts"])
    assert len(batch["evaluations"][0]["automatic"]["semantic_evaluation"]["calls"]) == len(facts) + 1
    assert len(calls) == len(facts) + 1


def test_null_reference_paraphrase_is_semantically_reviewed(client, monkeypatch):
    raw = '{"facts":[],"answer":"No answer can be determined from the supplied information."}'
    monkeypatch.setattr(
        "traceability_lab.pipeline.generate",
        lambda *args: {"choices": [{"message": {"content": raw}, "finish_reason": "stop"}]},
    )
    calls = []

    def reviewer(settings, body, chunks):
        calls.append(body)
        assert "Expected null accepts null or equivalent abstention" in body["messages"][0]["content"]
        assert "A textual reference requires its meaning" in body["messages"][0]["content"]
        assert "Evaluate the generated answer against the question" in body["messages"][0]["content"]
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "answer": {
                                    "meaning_match": "equivalent",
                                    "evidence_support": "supported",
                                    "confidence": 100,
                                    "explanation": "Equivalent absence of information.",
                                },
                                "unsupported_answer": False,
                            }
                        )
                    },
                    "finish_reason": "stop",
                }
            ]
        }

    monkeypatch.setattr("traceability_lab.semantic_evaluation.generate", reviewer)
    definition = quoted_experiment()
    definition["semantic_reference"] = {"facts": [], "answer": None}
    definition["semantic_evaluator"]["confidence_threshold"] = 100
    result = client.post(
        "/api/experiment-definitions/run",
        json={"definition": definition, "execution": {"repetitions": 1, "llm_profile_id": "test-generator"}},
    )
    assert result.status_code == 201, result.text
    batch = client.get(f"/api/batches/{result.json()['batch_id']}").json()
    assert len(calls) == 1
    assert all(count["pass"] == 1 for count in batch["property_counts"])


def test_exact_reference_keeps_source_and_quote_checks():
    from traceability_lab.fact_answers import exact_reference_match

    sources = [
        {"document_id": "a", "version": "1", "quote": "Evidence A"},
        {"document_id": "b", "version": "1", "quote": "Evidence B"},
    ]
    reference = {"facts": [{"fact": "Statement", "sources": sources}], "answer": "Answer"}
    answer = {"facts": [{"fact": "Statement", "sources": list(reversed(sources))}], "answer": "Answer"}
    run = {
        "backend": "local",
        "status": "completed",
        "answer": json.dumps(answer),
        "snapshot": {
            "evaluation_reference": {
                "definition": {"semantic_reference": reference, "answer_json": FACT_ANSWER_JSON}
            }
        },
        "context": [{"document_id": s["document_id"], "version": "1", "text": s["quote"]} for s in sources],
    }
    assert exact_reference_match(run)
    run["snapshot"]["context_edited"] = True
    assert not exact_reference_match(run)
    run["snapshot"]["context_edited"] = False
    run["context"][0]["text"] = "No supporting quotation here."
    assert not exact_reference_match(run)
    run["context"][0]["text"] = "Evidence A"
    run["context"][0]["document_id"] = "different-source"
    assert not exact_reference_match(run)


def test_answer_call_cannot_rejudge_individual_facts(client, monkeypatch):
    import traceability_lab.semantic_evaluation as semantic

    configure_calls(monkeypatch)
    original = semantic.generate

    def isolated(settings, body, chunks):
        payload = json.loads(body["messages"][1]["content"])
        if "generated_fact" not in payload:
            assert set(payload) == {"question", "expected_answer", "generated_answer", "documents"}
            assert "fact_judgments" not in payload
        return original(settings, body, chunks)

    monkeypatch.setattr(semantic, "generate", isolated)
    summary, observation = execute(client)
    assert observation["status"] == "completed", observation
    assert all(item["pass"] == 1 for item in summary["property_counts"])


def test_historical_joint_conflict_does_not_override_individually_approved_facts():
    from traceability_lab.fact_answers import fact_result_criteria

    definition = json.loads(Path("experiments/03-conflcting-sources.json").read_text())["experiment"]
    reference = definition["semantic_reference"]
    answer = {
        "facts": [{"fact": fact["fact"] + " ", "sources": fact["sources"]} for fact in reference["facts"]],
        "answer": None,
    }
    run = {
        "answer": json.dumps(answer),
        "status": "completed",
        "backend": "local",
        "snapshot": {"evaluation_reference": {"definition": {**definition, "answer_json": FACT_ANSWER_JSON}}},
        "context": [
            {
                "chunk_id": doc["document_id"],
                "document_id": doc["document_id"],
                "version": doc["version"],
                "text": doc["content"],
            }
            for doc in definition["documents"]
        ],
    }
    judgments = {
        "facts": [
            {
                "fact_index": i,
                "expected_fact_indices": [i],
                "meaning_match": "equivalent",
                "source_support": "supported",
                "confidence": 100,
                "explanation": "Correct report of this source.",
            }
            for i in range(len(answer["facts"]))
        ],
        "expected_facts": [
            {"expected_fact_index": i, "status": "covered", "confidence": 100}
            for i in range(len(answer["facts"]))
        ],
        "answer": {"meaning_match": "equivalent", "evidence_support": "supported", "confidence": 100},
        "unsupported_fact_indices": [],
        "conflicting_fact_indices": [0, 1],
        "unsupported_answer": False,
    }
    automatic = {
        "criteria": [{"id": "answer_json", "result": "pass"}],
        "semantic_evaluation": {
            "status": "completed",
            "configuration": {"confidence_threshold": 100},
            "judgments": judgments,
        },
    }
    before = deepcopy(automatic)
    criteria = fact_result_criteria(run, automatic)
    assert criteria[1]["result"] == "pass"
    assert criteria[2]["result"] == "fail"
    assert "null" in criteria[2]["reason"]
    assert automatic == before


def test_generator_and_reviewer_record_independent_temperatures(client, monkeypatch):
    calls = configure_calls(monkeypatch)
    definition = quoted_experiment()
    definition["semantic_evaluator"]["temperature"] = 0.2
    response = client.post(
        "/api/experiment-definitions/run",
        json={
            "definition": definition,
            "execution": {"repetitions": 1, "llm_profile_id": "test-generator", "temperature": 1.8},
        },
    )
    assert response.status_code == 201, response.text
    batch = client.get(f"/api/batches/{response.json()['batch_id']}").json()
    run = client.get(f"/api/runs/{batch['runs'][0]['id']}").json()
    assert run["snapshot"]["request_body"]["temperature"] == 1.8
    assert run["snapshot"]["question_configuration"]["temperature"] == 1.8
    observation = batch["evaluations"][0]["automatic"]["semantic_evaluation"]
    assert observation["configuration"]["temperature"] == 0.2
    assert len(calls) == 3
    assert all(call["temperature"] == 0.2 for call in calls)
    assert all(call["request_body"]["temperature"] == 0.2 for call in observation["calls"])
