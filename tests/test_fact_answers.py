"""Controlled facts mode: deterministic contracts and isolated optional model observations."""

import json

import pytest

from traceability_lab.db import json_text
from traceability_lab.fact_answers import FACT_ANSWER_JSON
from traceability_lab.models import ProfileInput
from traceability_lab.profiles import read_profile, save_profile


def experiment(client, *, evaluator=False):
    docs = client.get("/api/documents").json()
    versions = [
        next(d["id"] for d in docs if d["document_id"] == "orion" and d["version"] == version)
        for version in ("1", "2")
    ]
    return {
        "title": "Facts experiment",
        "answer_json": {
            "facts": [{"fact": "string", "sources": [{"document_id": "string", "version": "string"}]}],
            "answer": "string | null",
        },
        "question": "What does the evidence establish?",
        "version_ids": versions,
        "output_mode": "facts_and_answer",
        "system_prompt": "Researcher supplied policy.\n{{output_json}}",
        "context": "Research question: {{user_question}}\nSupplied evidence:\n{{documents}}",
        "semantic_reference": {
            "facts": ["EXPECTED PRIVATE REFERENCE"],
            "answer": "EXPECTED PRIVATE ANSWER",
        },
        "semantic_evaluator": {"llm_profile_id": "test-reviewer", "confidence_threshold": None}
        if evaluator
        else None,
    }


def generated(monkeypatch, make_answer):
    observed = []

    def generate(settings, body, chunks):
        assert "EXPECTED PRIVATE" not in json_text(body)
        assert "semantic_reference" not in json_text(body)
        assert "semantic_evaluator" not in json_text(body)
        assert body["messages"][0]["content"].startswith("Researcher supplied policy.")
        assert "OUTPUT CONTRACT" not in body["messages"][0]["content"]
        assert '"facts"' in body["messages"][0]["content"]
        observed.append(body)
        answer = make_answer(chunks)
        return {
            "choices": [
                {
                    "message": {"content": answer if isinstance(answer, str) else json.dumps(answer)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 20},
        }

    monkeypatch.setattr("traceability_lab.pipeline.generate", generate)
    return observed


def execute(client, data, **kwargs):
    response = client.post(
        "/api/experiment-definitions/run",
        json={
            "definition": data,
            "execution": {"repetitions": 1, "llm_profile_id": "test-generator", **kwargs},
        },
    )
    assert response.status_code == 201, response.text
    batch = client.get(f"/api/batches/{response.json()['batch_id']}").json()
    runs = [client.get(f"/api/runs/{item['id']}").json() for item in batch["runs"]]
    return batch, runs


@pytest.mark.parametrize(
    "kind", ["different_sources", "multiple_sources", "none", "insufficient", "conflict"]
)
def test_facts_can_vary_in_count_sources_and_answer(client, monkeypatch, kind):
    def answer(chunks):
        facts = [
            {"fact": "First statement.", "source_ids": [chunks[0]["id"]]},
            {"fact": "Different statement.", "source_ids": [chunks[-1]["id"]]},
        ]
        if kind == "multiple_sources":
            facts = [
                {"fact": "Comparison across documents.", "source_ids": [chunks[0]["id"], chunks[-1]["id"]]}
            ]
        if kind in ("none", "insufficient"):
            facts = []
        if kind == "conflict":
            facts[1]["fact"] = "The other source contradicts the first."
        return {
            "facts": facts,
            "answer": None if kind in ("none", "insufficient", "conflict") else "A free-text answer.",
        }

    observed = generated(monkeypatch, answer)
    data = experiment(client)
    preview = client.post(
        "/api/context-preview",
        json={
            "question": data["question"],
            "version_ids": data["version_ids"],
            "output_mode": "facts_and_answer",
            "context_mode": "explicit",
            "context_ids": [],
            "prompt_instructions": data["context"],
            "answer_json": data["answer_json"],
            "system_prompt": data["system_prompt"],
        },
    ).json()
    batch, runs = execute(
        client,
        data,
        system_prompt=preview["system_prompt"],
        llm_context=preview["text"],
        context_chunk_ids=preview["chunk_ids"],
    )
    assert runs[0]["messages"][-1]["content"] == preview["text"]
    assert runs[0]["snapshot"]["context_edited"] is False
    assert len(observed) == 1
    assert runs[0]["answer"] == json.dumps(answer([{**c, "id": c["chunk_id"]} for c in runs[0]["context"]]))
    assert [c["result"] for c in batch["evaluations"][0]["automatic"]["criteria"]] == ["pass", "pass"]
    assert "semantic_evaluation" not in batch["evaluations"][0]["automatic"]
    assert runs[0]["snapshot"]["experiment"]["definition"]["answer_json"] == FACT_ANSWER_JSON


@pytest.mark.parametrize(
    "answer",
    [
        "not json",
        '{"facts":[],"answer":null,"extra":1}',
        '{"facts":[]}',
        '{"facts":"wrong","answer":null}',
        '{"facts":[{"fact":"","source_ids":["x"]}],"answer":null}',
        '{"facts":[{"fact":"A fact","source_ids":[]}],"answer":null}',
        '{"facts":[],"answer":42}',
        '```json\n{"facts":[],"answer":null}\n```',
    ],
)
def test_invalid_fact_structure_is_preserved(client, monkeypatch, answer):
    generated(monkeypatch, lambda chunks: answer)
    batch, runs = execute(client, experiment(client))
    assert runs[0]["answer"] == answer
    checks = batch["evaluations"][0]["automatic"]["criteria"]
    assert checks[0]["result"] == "fail"
    assert checks[1]["result"] == "requires_review"


@pytest.mark.parametrize("source", ["nonexistent", "bracketed"])
def test_source_identifiers_must_be_exact_supplied_ids(client, monkeypatch, source):
    generated(
        monkeypatch,
        lambda chunks: {
            "facts": [
                {
                    "fact": "Statement",
                    "source_ids": [f"[{chunks[0]['id']}]" if source == "bracketed" else source],
                }
            ],
            "answer": None,
        },
    )
    batch, _ = execute(client, experiment(client))
    assert batch["evaluations"][0]["automatic"]["criteria"][1]["result"] == "fail"


def judgments():
    return {
        "facts": [
            {
                "fact_index": 0,
                "expected_fact_indices": [0],
                "meaning_match": "equivalent",
                "source_support": "supported",
                "explanation": "Equivalent paraphrase supported by cited passage.",
            }
        ],
        "expected_facts": [
            {"expected_fact_index": 0, "status": "covered", "explanation": "Expressed in other words."}
        ],
        "answer": {
            "meaning_match": "equivalent",
            "evidence_support": "supported",
            "explanation": "Same meaning.",
        },
        "unsupported_fact_indices": [],
        "conflicting_fact_indices": [],
        "unsupported_answer": False,
    }


def test_optional_evaluator_is_separate_frozen_and_traceable(client, monkeypatch, db):
    calls = []
    generator_calls = generated(
        monkeypatch,
        lambda chunks: {
            "facts": [{"fact": "A differently worded observation.", "source_ids": [chunks[0]["id"]]}],
            "answer": "Rephrased answer.",
        },
    )
    data = experiment(client, evaluator=True)

    def evaluate(settings, body, chunks):
        assert settings.llm_model == "test-reviewer-model"
        payload = json.loads(body["messages"][1]["content"])
        if "generated_fact" in payload:
            assert (
                payload["candidate_expected_facts"][0]["expected_fact"]
                == data["semantic_reference"]["facts"][0]
            )
            assert payload["supplied_sources"][0]["sha256"]
        else:
            assert payload["expected_answer"] == data["semantic_reference"]["answer"]
        assert chunks == []
        assert "generator_request" not in payload
        calls.append(body)
        # Changing a profile mid-batch must not change subsequent evaluator settings.
        profile = read_profile(db, "test-reviewer")
        save_profile(
            db,
            ProfileInput(
                **{
                    **{k: v for k, v in profile.items() if k not in {"id", "created_at", "updated_at"}},
                    "model": "later-model",
                }
            ),
            "test-reviewer",
        )
        value = (
            judgments()["facts"][0]
            if "generated_fact" in payload
            else {key: judgments()[key] for key in ("answer", "unsupported_answer")}
        )
        return {"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]}

    monkeypatch.setattr("traceability_lab.semantic_evaluation.generate", evaluate)
    batch, runs = execute(client, data, repetitions=2)
    assert len(generator_calls) == 2
    assert len(calls) == 4
    for index, evaluation in enumerate(batch["evaluations"]):
        observation = evaluation["automatic"]["semantic_evaluation"]
        assert observation["status"] == "completed"
        assert observation["judgments"]["facts"] == [
            {**fact, "correctness_score": None, "confidence": None} for fact in judgments()["facts"]
        ]
        assert observation["judgments"]["answer"] == {
            **judgments()["answer"],
            "correctness_score": None,
            "confidence": None,
        }
        assert observation["request_body"] == calls[index * 2 + 1]
        assert observation["raw_answer"] == json.dumps(
            {key: judgments()[key] for key in ("answer", "unsupported_answer")}
        )
        assert runs[index]["answer"] == json.dumps(
            {
                "facts": [
                    {
                        "fact": "A differently worded observation.",
                        "source_ids": [runs[index]["context"][0]["chunk_id"]],
                    }
                ],
                "answer": "Rephrased answer.",
            }
        )
        spans = client.get(f"/api/runs/{runs[index]['id']}/trace").json()
        assert any(
            span["name"] == "evaluation.semantic" and span["span_id"] == observation["calls"][-1]["span_id"]
            for span in spans
        )
        assert observation["trace_id"] == runs[index]["trace_id"]
    frozen = batch["evaluations"][0]["automatic"]
    data["semantic_reference"]["facts"] = ["Changed later"]
    client.post("/api/experiment-definitions", json=data)
    assert client.get(f"/api/runs/{runs[0]['id']}/evaluation").json()["automatic"] == frozen


@pytest.mark.parametrize("failure", ["invalid", "offline", "indices"])
def test_evaluator_failure_never_changes_generator(client, monkeypatch, failure):
    generated(monkeypatch, lambda chunks: {"facts": [], "answer": None})

    def evaluate(*args):
        if failure == "offline":
            raise TimeoutError("Evaluator unavailable")
        raw = "not json" if failure == "invalid" else json.dumps(judgments())
        return {"choices": [{"message": {"content": raw}, "finish_reason": "stop"}]}

    monkeypatch.setattr("traceability_lab.semantic_evaluation.generate", evaluate)
    batch, runs = execute(client, experiment(client, evaluator=True))
    assert runs[0]["status"] == "completed"
    assert runs[0]["answer"] == '{"facts": [], "answer": null}'
    assert batch["evaluations"][0]["automatic"]["semantic_evaluation"]["status"] == "failed"


def test_invalid_generator_skips_optional_evaluator(client, monkeypatch):
    generated(monkeypatch, lambda chunks: "invalid")
    monkeypatch.setattr(
        "traceability_lab.semantic_evaluation.generate", lambda *args: pytest.fail("Evaluator must not run")
    )
    batch, _ = execute(client, experiment(client, evaluator=True))
    assert batch["evaluations"][0]["automatic"]["semantic_evaluation"]["status"] == "not_evaluated"


@pytest.mark.parametrize(
    "context", ["", "My reusable instructions.", "Question: {{user_question}}", "{{documents}}"]
)
def test_fact_preview_and_execution_match_without_required_wildcards(client, monkeypatch, context):
    data = experiment(client)
    data["context"] = context
    calls = generated(monkeypatch, lambda chunks: {"facts": [], "answer": None})
    preview = client.post(
        "/api/context-preview",
        json={
            "question": data["question"],
            "version_ids": data["version_ids"],
            "context_mode": "explicit",
            "context_ids": [],
            "output_mode": "facts_and_answer",
            "prompt_instructions": context,
            "answer_json": data["answer_json"],
            "system_prompt": data["system_prompt"],
        },
    ).json()
    _, direct = execute(client, data)
    _, prepared = execute(
        client,
        data,
        system_prompt=preview["system_prompt"],
        llm_context=preview["text"],
        context_chunk_ids=preview["chunk_ids"],
    )
    assert direct[0]["messages"] == prepared[0]["messages"]
    assert direct[0]["messages"][-1]["content"] == preview["text"]
    assert preview["text"].count("### BEGIN DOCUMENT") == 2
    assert preview["text"].count("### END DOCUMENT") == 2
    for source in direct[0]["context"]:
        assert source["chunk_id"] not in preview["text"]
    assert data["question"] in preview["text"]
    assert 'Source reference: {"document_id": "orion", "version": "1"}' in preview["text"]
    assert direct[0]["messages"][0]["content"].count("OUTPUT CONTRACT") == 0
    assert len(calls) == 2


def test_evaluator_without_expected_reference_can_assess_support_only(client, monkeypatch):
    generated(monkeypatch, lambda chunks: {"facts": [], "answer": None})
    data = experiment(client, evaluator=True)
    data["semantic_reference"] = None

    def evaluate(settings, body, chunks):
        assert json.loads(body["messages"][1]["content"])["expected_answer"] is None
        judgment = {
            "facts": [],
            "expected_facts": [],
            "answer": {
                "meaning_match": "not_evaluated",
                "evidence_support": "insufficient_evidence",
                "explanation": "No answer was reached.",
            },
            "unsupported_fact_indices": [],
            "conflicting_fact_indices": [],
            "unsupported_answer": False,
        }
        for key in ("facts", "expected_facts", "unsupported_fact_indices", "conflicting_fact_indices"):
            judgment.pop(key)
        return {"choices": [{"message": {"content": json.dumps(judgment)}, "finish_reason": "stop"}]}

    monkeypatch.setattr("traceability_lab.semantic_evaluation.generate", evaluate)
    batch, _ = execute(client, data)
    assert batch["evaluations"][0]["automatic"]["semantic_evaluation"]["status"] == "completed"


def test_failed_generator_skips_semantics_and_preserves_failure(client, monkeypatch):
    data = experiment(client, evaluator=True)

    def fail(*args):
        raise TimeoutError("Generator failed")

    monkeypatch.setattr("traceability_lab.pipeline.generate", fail)
    monkeypatch.setattr(
        "traceability_lab.semantic_evaluation.generate", lambda *args: pytest.fail("Not called")
    )
    batch, runs = execute(client, data)
    assert runs[0]["status"] == "failed"
    assert batch["evaluations"][0]["automatic"]["semantic_evaluation"]["status"] == "not_evaluated"


def test_conflicting_missing_and_unsupported_judgments_stay_separate(client, monkeypatch):
    generated(
        monkeypatch,
        lambda chunks: {
            "facts": [{"fact": "An unsupported inference.", "source_ids": [chunks[0]["id"]]}],
            "answer": "A questionable answer.",
        },
    )
    judgment = judgments()
    judgment["facts"][0].update(
        expected_fact_indices=[],
        meaning_match="not_equivalent",
        source_support="contradicted",
        explanation="Conflicts with the cited passage.",
    )
    judgment["expected_facts"][0].update(status="missing", explanation="Important expected fact omitted.")
    judgment["answer"].update(meaning_match="not_equivalent", evidence_support="insufficient_evidence")
    judgment.update(unsupported_fact_indices=[0], conflicting_fact_indices=[0], unsupported_answer=True)

    def reviewer(settings, body, chunks):
        payload = json.loads(body["messages"][1]["content"])
        value = (
            judgment["facts"][0]
            if "generated_fact" in payload
            else {key: judgment[key] for key in ("answer", "unsupported_answer")}
        )
        return {"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]}

    monkeypatch.setattr("traceability_lab.semantic_evaluation.generate", reviewer)
    batch, _ = execute(client, experiment(client, evaluator=True))
    automatic = batch["evaluations"][0]["automatic"]
    assert automatic["criteria"][1]["result"] == "pass"  # Identity check only.
    recorded = automatic["semantic_evaluation"]["judgments"]
    assert recorded["facts"] == [
        {**fact, "correctness_score": None, "confidence": None} for fact in judgment["facts"]
    ]
    assert recorded["expected_facts"][0]["status"] == "missing"
    assert recorded["unsupported_fact_indices"] == [0]
    assert recorded["conflicting_fact_indices"] == []
    assert recorded["answer"] == {**judgment["answer"], "correctness_score": None, "confidence": None}
    assert "score" not in automatic["semantic_evaluation"]["judgments"]


def test_evaluator_profile_is_validated_before_generator_runs(client, monkeypatch):
    data = experiment(client, evaluator=True)
    data["semantic_evaluator"]["llm_profile_id"] = "test-embedding"
    monkeypatch.setattr(
        "traceability_lab.pipeline.generate", lambda *args: pytest.fail("Generator must not start")
    )
    response = client.post("/api/experiment-definitions/run", json={"definition": data})
    assert response.status_code == 422
    assert client.get("/api/runs").json() == []


def test_edited_context_cannot_acquire_verified_source_support(client, monkeypatch):
    generated(
        monkeypatch,
        lambda chunks: {"facts": [{"fact": "Claim", "source_ids": ["unknown"]}], "answer": "Claim"},
    )
    monkeypatch.setattr(
        "traceability_lab.semantic_evaluation.generate",
        lambda *args: {
            "choices": [{"message": {"content": json.dumps(judgments())}, "finish_reason": "stop"}]
        },
    )
    batch, runs = execute(
        client, experiment(client, evaluator=True), llm_context="Edited evidence", context_chunk_ids=[]
    )
    assert runs[0]["snapshot"]["context_edited"]
    assert batch["evaluations"][0]["automatic"]["criteria"][1]["result"] == "requires_review"
    assert batch["evaluations"][0]["automatic"]["semantic_evaluation"]["status"] == "failed"


def test_document_reference_contract_keeps_expected_answer_private(client, monkeypatch):
    from pathlib import Path

    data = json.loads(Path("experiments/01-complete-experiment.json").read_text())["experiment"]
    data["semantic_evaluator"] = None
    data["semantic_reference"]["facts"][0]["fact"] = "EXPECTED PRIVATE FACT"
    calls = generated(
        monkeypatch,
        lambda chunks: {
            "facts": [
                {
                    "fact": "Anna manages Orion.",
                    "sources": [
                        {"document_id": "orion", "version": "1", "quote": "Project manager: Anna de Vries."}
                    ],
                }
            ],
            "answer": "Anna manages Orion.",
        },
    )
    data["system_prompt"] = "Researcher supplied policy.\n{{output_json}}"
    batch, runs = execute(client, data)
    assert all(item["result"] == "pass" for item in batch["evaluations"][0]["automatic"]["criteria"])
    system = calls[0]["messages"][0]["content"]
    assert '"document_id"' in system and '"version"' in system
    assert '"quote"' in system
    assert "EXPECTED PRIVATE" not in json_text(calls)
    assert runs[0]["snapshot"]["evaluation_reference"]["definition"]["semantic_reference"]["facts"][0][
        "sources"
    ] == [{"document_id": "orion", "version": "1", "quote": "Project manager: Anna de Vries."}]


@pytest.mark.parametrize(
    ("optional", "coverage", "result"),
    [
        (True, "missing", "not_applicable"),
        (False, "missing", "fail"),
        (True, "covered", "pass"),
        (True, "partial", "fail"),
        (True, "not_evaluated", "requires_review"),
    ],
)
def test_optional_reference_coverage(optional, coverage, result):
    from traceability_lab.semantic_evaluation import reference_checks

    value = judgments()
    value["facts"] = []
    value["expected_facts"][0]["status"] = coverage
    checked = reference_checks(value, {"facts": [{"fact": "Expected", "optional": optional}]})
    assert checked["expected_facts"][0]["result"] == result
    assert checked["expected_facts"][0]["optional"] is optional


@pytest.mark.parametrize(
    ("indices", "meaning", "support", "result"),
    [
        ([], "not_equivalent", "supported", "fail"),
        ([0], "equivalent", "supported", "pass"),
        ([0], "partial", "supported", "fail"),
        ([0], "equivalent", "contradicted", "fail"),
        ([0], "equivalent", "not_evaluated", "requires_review"),
    ],
)
def test_generated_reference_membership_and_support(indices, meaning, support, result):
    from traceability_lab.semantic_evaluation import reference_checks

    value = judgments()
    value["facts"][0].update(expected_fact_indices=indices, meaning_match=meaning, source_support=support)
    checked = reference_checks(value, {"facts": [{"fact": "Expected", "optional": True}]})
    assert checked["generated_facts"][0]["result"] == result


@pytest.mark.parametrize("include_optional", [False, True])
def test_optional_facts_are_frozen_private_unordered_and_reported(client, monkeypatch, include_optional):
    from copy import deepcopy

    data = experiment(client, evaluator=True)
    source = {"document_id": "orion", "version": "1", "quote": "Anna de Vries"}
    data["semantic_reference"]["facts"] = [
        {"fact": "EXPECTED PRIVATE required", "sources": [source]},
        {"fact": "EXPECTED PRIVATE optional", "sources": [source], "optional": True},
    ]
    # Optional fact deliberately precedes required fact in the generated answer.
    generated(
        monkeypatch,
        lambda chunks: {
            "facts": ([{"fact": "Additional allowed fact", "sources": [source]}] if include_optional else [])
            + [{"fact": "Essential fact", "sources": [source]}],
            "answer": "Answer",
        },
    )

    def evaluate(settings, body, chunks):
        payload = json.loads(body["messages"][1]["content"])
        if "generated_fact" in payload:
            index = payload["fact_index"]
            value = {
                **deepcopy(judgments()["facts"][0]),
                "fact_index": index,
                "expected_fact_indices": [1 if include_optional and index == 0 else 0],
            }
            assert len(payload["candidate_expected_facts"]) == 2
            return {"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]}
        assert "expected_reference" not in payload
        assert payload["expected_answer"] == data["semantic_reference"]["answer"]
        value = judgments()
        value["facts"] = [
            {
                **deepcopy(judgments()["facts"][0]),
                "fact_index": i,
                "expected_fact_indices": [1 if include_optional and i == 0 else 0],
            }
            for i in range(2 if include_optional else 1)
        ]
        value["expected_facts"].append(
            {
                "expected_fact_index": 1,
                "status": "covered" if include_optional else "missing",
                "explanation": "Optional coverage.",
            }
        )
        for key in ("facts", "expected_facts", "unsupported_fact_indices", "conflicting_fact_indices"):
            value.pop(key)
        return {"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]}

    monkeypatch.setattr("traceability_lab.semantic_evaluation.generate", evaluate)
    batch, runs = execute(client, data)
    observation = batch["evaluations"][0]["automatic"]["semantic_evaluation"]
    assert observation["status"] == "completed", observation
    assert observation["prompt_version"] == "facts-semantic-v15"
    assert [item["id"] for item in batch["property_counts"]] == ["answer_json", "facts", "answer"]
    assert all(item["pass"] == 1 for item in batch["property_counts"])
    assert batch["evaluations"][0]["result_criteria_version"] == "facts-results-v8"
    assert [item["id"] for item in batch["evaluations"][0]["automatic"]["criteria"]] == [
        "answer_json",
        "source_ids",
    ]
    checks = observation["reference_checks"]
    assert checks["expected_facts"][1]["result"] == ("pass" if include_optional else "not_applicable")
    assert all(check["result"] == "pass" for check in checks["generated_facts"])
    assert (
        runs[0]["snapshot"]["evaluation_reference"]["definition"]["semantic_reference"]["facts"][1][
            "optional"
        ]
        is True
    )
    saved = client.post("/api/experiment-definitions", json=data).json()
    exported = client.get(f"/api/experiment-definitions/{saved['id']}/export").json()
    imported = client.post("/api/experiment-definitions/import", json=exported)
    assert imported.status_code == 200, imported.text
    assert imported.json()["definition"]["semantic_reference"]["facts"][1]["optional"] is True


@pytest.mark.parametrize(
    "scenario",
    [
        "correct",
        "wrong_source",
        "unknown_source",
        "unsupported",
        "missing_required",
        "missing_optional",
        "unexpected",
        "wrong_answer",
        "no_evaluator",
        "unknown_support",
        "null_answer",
    ],
)
def test_answer_properties_combine_fact_meaning_and_expected_sources(scenario):
    from copy import deepcopy

    from traceability_lab.fact_answers import fact_result_criteria

    source = {"document_id": "orion", "version": "1"}
    reference = {"facts": [{"fact": "Expected", "sources": [source]}], "answer": "Expected answer"}
    answer = {
        "facts": [{"fact": "Equivalent wording", "sources": [source]}],
        "answer": "Equivalent answer",
    }
    value = judgments()
    automatic = {
        "criteria": [
            {"id": "answer_json", "result": "pass", "method": "deterministic", "applicability": "applicable"}
        ],
        "semantic_evaluation": {"status": "completed", "judgments": value},
    }
    if scenario == "wrong_source":
        answer["facts"][0]["sources"] = [{"document_id": "orion", "version": "2"}]
    if scenario == "unknown_source":
        answer["facts"][0]["sources"] = [{"document_id": "internal-hash", "version": "1"}]
    if scenario == "unsupported":
        value["facts"][0]["source_support"] = "insufficient_evidence"
    if scenario in {"missing_required", "missing_optional"}:
        reference["facts"].append(
            {"fact": "Another", "sources": [source], "optional": scenario == "missing_optional"}
        )
        value["expected_facts"].append(
            {"expected_fact_index": 1, "status": "missing", "explanation": "Absent"}
        )
    if scenario == "unexpected":
        value["facts"][0].update(expected_fact_indices=[], meaning_match="not_equivalent")
        value["expected_facts"][0]["status"] = "missing"
    if scenario == "wrong_answer":
        value["answer"]["meaning_match"] = "not_equivalent"
    if scenario == "no_evaluator":
        automatic.pop("semantic_evaluation")
    if scenario == "unknown_support":
        value["facts"][0]["source_support"] = "not_evaluated"
    if scenario == "null_answer":
        answer["answer"] = reference["answer"] = None
        value["answer"]["evidence_support"] = "not_evaluated"
    run = {
        "answer": json.dumps(answer),
        "backend": "local",
        "status": "completed",
        "snapshot": {"evaluation_reference": {"definition": {"semantic_reference": reference}}},
        "context": [
            {"chunk_id": "chunk1", "document_id": "orion", "version": "1"},
            {"chunk_id": "chunk2", "document_id": "orion", "version": "2"},
        ],
    }
    before = deepcopy(automatic)
    results = {item["id"]: item for item in fact_result_criteria(run, automatic)}
    assert list(results) == ["answer_json", "facts", "answer"]
    assert results["facts"]["result"] == (
        "fail"
        if scenario in {"wrong_source", "unknown_source", "unsupported", "missing_required", "unexpected"}
        else "fail"
        if scenario in {"no_evaluator", "unknown_support"}
        else "pass"
    )
    assert results["answer"]["result"] == (
        "fail" if scenario == "wrong_answer" else "fail" if scenario == "no_evaluator" else "pass"
    )
    assert results["facts"]["observed"] == answer["facts"]
    assert results["facts"]["expected"] == reference["facts"]
    assert automatic == before
