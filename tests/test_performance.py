"""Metrics use recorded intervals and matched token/time populations only."""

import json
from copy import deepcopy

import pytest

from traceability_lab.performance import experiment_overview, provider_metrics, run_performance, speed

START = "2026-01-01T00:00:00+00:00"
END = "2026-01-01T00:00:10+00:00"


def record(identity="one", duration=1000, *, status="completed", output=100, decode=1000, cache=20):
    response = {
        "usage": {"prompt_tokens": 200, "completion_tokens": output, "total_tokens": 200 + output},
        "timings": {"predicted_ms": decode, "predicted_n": output, "predicted_per_second": 99999},
    }
    if cache is not None:
        response["usage"]["prompt_tokens_details"] = {"cached_tokens": cache}
    run = {
        "id": identity,
        "status": status,
        "backend": "local",
        "created_at": START,
        "finished_at": END,
        "response": response,
        "snapshot": {
            "model": {"id": "frozen-model"},
            "request_body": {"temperature": 0.2, "seed": 7},
            "profiles": {"llm": {"name": "Frozen model name"}},
            "experiment": {
                "definition": {
                    "documents": [
                        {"document_id": "orion", "version": "1", "enabled": True},
                        {"document_id": "hidden", "version": "1", "enabled": False},
                    ]
                }
            },
        },
        "context": [{"text": "abc", "chunk_id": "a"}, {"text": "def", "chunk_id": "b"}],
    }
    trace = (
        []
        if duration is None
        else [
            {
                "name": "llm.generate",
                "start_ns": 1000000,
                "end_ns": 1000000 + duration * 1000000,
                "duration_ms": duration,
            }
        ]
    )
    return run_performance(run, trace, None)


def test_completed_extremes_average_wall_clock_and_weighted_decode():
    rows = [record(duration=1000, output=100, decode=1000), record("two", 3000, output=20, decode=100)]
    before = deepcopy(rows)
    result = experiment_overview(rows, {"started_at": START, "finished_at": END})
    assert result["total_duration_ms"] == 10000
    assert result["average_generator_ms"] == 2000
    assert result["fastest"]["run_id"] == "one"
    assert result["slowest"]["run_id"] == "two"
    assert result["generator"]["decode_tokens_per_second"] == pytest.approx(120 / 1.1)
    assert result["generator"]["end_to_end_tokens_per_second"] == 30
    assert result["configuration"]["unique_enabled_document_versions"] == 1
    assert result["configuration"]["source_fragments"] == {"min": 2, "max": 2, "total": 4, "measured_runs": 2}
    assert result["configuration"]["source_text_characters"]["total"] == 12
    assert result["configuration"]["models"][0]["name"] == "Frozen model name"
    assert result["configuration"]["models"][0]["identity"]["technical"]["api_model_id"] == "frozen-model"
    assert result["configuration"]["models"][0]["id"] is None
    assert rows == before


def test_failed_running_and_missing_times_do_not_win_or_become_zero():
    rows = [
        record(),
        record("failed", 1, status="failed"),
        record("running", None, status="running"),
        record("old", None),
    ]
    rows[1]["response"] = None
    result = experiment_overview(rows, {"started_at": START, "finished_at": None})
    assert result["total_duration_ms"] is None
    assert result["average_generator_ms"] == 1000
    assert result["duration_measured_runs"] == 1
    assert result["completed_generator_runs"] == 2
    assert result["fastest"]["run_id"] == result["slowest"]["run_id"] == "one"
    assert result["failed_runs"] == 1
    assert result["generator"]["tokens"]["input_tokens"] is None
    assert result["generator"]["tokens"]["reported"]["input_tokens"] == 3


def test_cache_is_subset_and_missing_cache_stays_unknown():
    result = experiment_overview([record(), record("two")])
    tokens = result["generator"]["tokens"]
    assert tokens["input_tokens"] == 400
    assert tokens["cached_input_tokens"] == 40
    assert tokens["output_tokens"] == 200
    assert tokens["total_tokens"] == 600
    result = experiment_overview([record(cache=None)])
    assert result["generator"]["tokens"]["cached_input_tokens"] is None
    assert provider_metrics({"timings": {"cache_n": 0}})["cached_input_tokens"] == 0


def test_evaluator_tokens_durations_and_errors_are_separate():
    row = record()
    row["evaluator"] = {
        "configured": True,
        "model": {"id": "evaluator"},
        "duration_basis": "recorded calls",
        "calls": [
            {
                "duration_ms": 2000,
                "status": "completed",
                "response": {"usage": {"prompt_tokens": 40, "completion_tokens": 10}},
            },
            {
                "duration_ms": 1000,
                "status": "failed",
                "response": {"usage": {"prompt_tokens": 60, "completion_tokens": 20}},
            },
        ],
    }
    result = experiment_overview([row])
    assert result["generator"]["tokens"]["total_tokens"] == 300
    assert result["evaluator"]["tokens"]["total_tokens"] == 130
    assert result["evaluator"]["calls"] == 2
    assert result["evaluator"]["total_duration_ms"] == 3000
    assert result["evaluator"]["average_duration_ms"] == 1500
    assert result["evaluator"]["errors"] == 1


def test_concurrent_runs_use_elapsed_batch_interval_not_summed_work():
    rows = [record(duration=10000), record("overlap", 5000)]
    result = experiment_overview(rows, {"started_at": START, "finished_at": END})
    assert result["total_duration_ms"] == 10000
    assert sum(row["generator_duration_ms"] for row in result["runs"]) == 15000
    assert result["average_generator_ms"] == 7500


def test_empty_single_and_historical_records():
    empty = experiment_overview([])
    assert empty["completed_runs"] == empty["failed_runs"] == 0
    assert empty["average_generator_ms"] is None and empty["total_duration_ms"] is None
    assert empty["fastest"] is None and empty["slowest"] is None
    assert empty["generator"]["tokens"]["total_tokens"] is None
    single = experiment_overview([record()])
    assert single["fastest"]["run_id"] == single["slowest"]["run_id"] == "one"
    assert single["average_generator_ms"] == 1000
    old = run_performance({"id": "old", "status": "completed", "backend": "local", "snapshot": {}}, [], None)
    result = experiment_overview([old])
    assert result["average_generator_ms"] is None
    assert result["configuration"]["unique_enabled_document_versions"] is None
    assert result["configuration"]["source_fragments"]["total"] is None


def test_only_explicit_provider_ttft_and_decode_times_are_used():
    assert provider_metrics({"timings": {"prompt_ms": 200}})["ttft_ms"] is None
    result = speed(
        [{"response": {"usage": {"completion_tokens": 10}, "timings": {"ttft_ms": 5}}, "duration_ms": 1000}]
    )
    assert result["decode_tokens_per_second"] is None
    assert result["end_to_end_tokens_per_second"] == 10
    assert result["average_ttft_ms"] == 5
    for value in [-1, float("nan"), float("inf"), True]:
        assert provider_metrics({"usage": {"completion_tokens": value}})["output_tokens"] is None


def test_legacy_evaluator_and_disabled_evaluator():
    run = {"id": "old", "status": "completed", "snapshot": {}}
    observation = {
        "request_body": {"model": "eval"},
        "duration_ms": 25,
        "status": "completed",
        "response": {"usage": {"prompt_tokens": 3, "completion_tokens": 2}},
    }
    result = experiment_overview([run_performance(run, [], {"semantic_evaluation": observation})])
    assert result["evaluator"]["calls"] == 1
    assert result["evaluator"]["duration_bases"] == ["legacy evaluation interval"]
    assert result["evaluator"]["tokens"]["total_tokens"] == 5
    assert experiment_overview([record()])["evaluator"]["configured"] is False


def test_stream_summary_contains_finished_batch_interval(client):
    response = client.post(
        "/api/experiment-definitions/stream",
        json={
            "definition": {
                "title": "Empty sources",
                "question": "Question?",
                "documents": [],
                "output_mode": "facts_and_answer",
            }
        },
    )
    events = [json.loads(line) for line in response.text.splitlines() if line]
    overview = events[-1]["summary"]["overview"]
    assert events[-1]["type"] == "completed"
    assert overview["total_duration_ms"] is not None
    assert overview["configuration"]["unique_enabled_document_versions"] == 0
    assert overview["configuration"]["source_fragments"]["total"] == 0
    assert overview["average_generator_ms"] is None  # Demo isn't a measured model.


def test_zero_tokens_are_valid_and_malformed_provider_metadata_is_unknown():
    row = record(output=0)
    row["response"] = {
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "prompt_tokens_details": {"cached_tokens": 0}}
    }
    result = experiment_overview([row])
    assert result["generator"]["tokens"]["total_tokens"] == 0
    assert result["generator"]["tokens"]["cached_input_tokens"] == 0
    for response in [
        None,
        {},
        {"usage": "unknown", "timings": []},
        {"usage": {"prompt_tokens_details": False}},
    ]:
        assert provider_metrics(response)["input_tokens"] is None
    assert provider_metrics({"timings": {"ttft_ms": None, "time_to_first_token_ms": 12}})["ttft_ms"] == 12


def test_selected_context_without_a_recorded_generator_request_is_not_reported_as_supplied():
    run = {
        "id": "failed-before-model",
        "status": "failed",
        "backend": "local",
        "snapshot": {},
        "context": [{"text": "selected but never sent"}],
    }
    row = run_performance(run, [], None)
    assert row["source_fragments"] is None
    assert row["source_text_characters"] is None
