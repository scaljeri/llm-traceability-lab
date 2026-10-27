"""Performance projections from recorded requests, provider values and trace boundaries."""

import math
from datetime import datetime
from statistics import mean

from traceability_lab.model_identity import recorded_identity


def number(value):
    return (
        value
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
        else None
    )


def elapsed(start, end):
    try:
        return number((datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() * 1000)
    except TypeError, ValueError:
        return None


def provider_metrics(response):
    response = response if isinstance(response, dict) else {}
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    timings = response.get("timings") if isinstance(response.get("timings"), dict) else {}
    details = (
        usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    )
    incoming, outgoing = number(usage.get("prompt_tokens")), number(usage.get("completion_tokens"))
    cached = number(details.get("cached_tokens"))
    if cached is None:
        cached = number(timings.get("cache_n"))
    decode_ms = number(timings.get("predicted_ms"))
    decode_tokens = number(timings.get("predicted_n"))
    if decode_tokens is None:
        decode_tokens = outgoing
    ttft = number(timings.get("ttft_ms"))
    if ttft is None:
        ttft = number(timings.get("time_to_first_token_ms"))
    return {
        "input_tokens": incoming,
        "output_tokens": outgoing,
        "total_tokens": incoming + outgoing if incoming is not None and outgoing is not None else None,
        "cached_input_tokens": cached,
        "decode_ms": decode_ms,
        "decode_tokens": decode_tokens,
        "ttft_ms": ttft,
    }


def token_summary(calls):
    metrics = [provider_metrics(call.get("response")) for call in calls]
    result = {"calls": len(calls), "reported": {}}
    for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens"):
        values = [item[key] for item in metrics if item[key] is not None]
        result[key] = sum(values) if calls and len(values) == len(calls) else None
        result["reported"][key] = len(values)
    return result


def speed(calls):
    pairs = [(provider_metrics(call.get("response")), call.get("duration_ms")) for call in calls]
    decoded = [
        item
        for item, _ in pairs
        if item["decode_ms"] is not None and item["decode_ms"] > 0 and item["decode_tokens"] is not None
    ]
    end_to_end = [
        (item["output_tokens"], duration)
        for item, duration in pairs
        if item["output_tokens"] is not None and duration is not None and duration > 0
    ]
    ttft = [item["ttft_ms"] for item, _ in pairs if item["ttft_ms"] is not None]
    return {
        "calls": len(calls),
        "decode_tokens_per_second": 1000
        * sum(item["decode_tokens"] for item in decoded)
        / sum(item["decode_ms"] for item in decoded)
        if decoded
        else None,
        "decode_measured_calls": len(decoded),
        "decode_ms": sum(item["decode_ms"] for item in decoded) if decoded else None,
        "end_to_end_tokens_per_second": 1000
        * sum(tokens for tokens, _ in end_to_end)
        / sum(duration for _, duration in end_to_end)
        if end_to_end
        else None,
        "end_to_end_measured_calls": len(end_to_end),
        "average_ttft_ms": mean(ttft) if ttft else None,
        "ttft_measured_calls": len(ttft),
    }


def run_performance(run, trace, automatic):
    snapshot = run.get("snapshot") or {}
    spans = [
        span
        for span in trace
        if span["name"] in {"llm.generate", "llm.repair_json"} and number(span.get("duration_ms")) is not None
    ]
    duration = (
        (max(span["end_ns"] for span in spans) - min(span["start_ns"] for span in spans)) / 1_000_000
        if spans
        else None
    )
    if run.get("backend") == "demo":
        duration = None
    observation = (automatic or {}).get("semantic_evaluation") or {}
    calls = observation.get("calls")
    if calls is None:
        calls = [observation] if observation.get("request_body") else []
    config = snapshot.get("question_configuration") or {}
    request = snapshot.get("request_body") or {}
    definition = (snapshot.get("experiment") or {}).get("definition") or {}
    documents = definition.get("documents")
    enabled = (
        [
            {"document_id": doc["document_id"], "version": str(doc["version"])}
            for doc in documents
            if doc.get("enabled", True)
        ]
        if documents is not None
        else [
            {"document_id": doc["document_id"], "version": str(doc["version"])}
            for doc in snapshot.get("corpus", [])
        ]
    )
    request_recorded = bool(request.get("messages") or run.get("messages") or spans)
    context_known = "context" in run and not snapshot.get("context_edited") and request_recorded
    source_context = run.get("context") or []
    profile = (snapshot.get("profiles") or {}).get("llm") or {}
    generator_identity = recorded_identity(snapshot.get("model"), profile)
    reviewer_identity = recorded_identity(observation.get("model"), observation.get("profile"))
    evaluation_end = observation.get("finished_at") or run.get("finished_at")
    repair = snapshot.get("json_repair")
    original_response = repair["original_response"] if repair else run.get("response")
    original_spans = [span for span in spans if span["name"] == "llm.generate"]
    original_ms = sum(span["duration_ms"] for span in original_spans) if original_spans else duration
    generator_calls = [{"response": original_response, "duration_ms": original_ms}]
    if repair:
        generator_calls.append({"response": repair.get("response"), "duration_ms": repair.get("duration_ms")})
    for call in generator_calls:
        call["response"] = {
            key: (call.get("response") or {})[key]
            for key in ("usage", "timings")
            if key in (call.get("response") or {})
        }
    totals = token_summary(generator_calls)
    return {
        "run_id": run["id"],
        "status": run["status"],
        "backend": run.get("backend"),
        "generator_duration_ms": duration,
        "full_run_duration_ms": elapsed(run.get("created_at"), evaluation_end),
        "generator_tokens": {
            key: totals[key]
            for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens")
        },
        **({"generator_calls": generator_calls} if repair else {}),
        "json_format": {
            "initial_valid": snapshot.get("initial_json_format_valid"),
            "repair_status": repair.get("status") if repair else None,
        },
        "generator_speed": speed(generator_calls),
        "response": {
            key: (run.get("response") or {})[key]
            for key in ("usage", "timings")
            if key in (run.get("response") or {})
        },
        "started_at": run.get("created_at"),
        "finished_at": evaluation_end,
        "model": {
            "id": generator_identity["profile_id"],
            "name": generator_identity["display_name"],
            "identity": generator_identity,
        },
        "temperature": request.get("temperature", config.get("temperature")),
        "seed": request.get("seed", config.get("seed")),
        "enabled_documents": enabled if documents is not None or "corpus" in snapshot else None,
        "source_fragments": len(source_context) if context_known else None,
        "source_text_characters": sum(len(item["text"]) for item in source_context)
        if context_known
        else None,
        "source_context_tokens": None,
        "evaluator": {
            "configured": bool(definition.get("semantic_evaluator") or observation),
            "model": {
                "id": reviewer_identity["profile_id"],
                "name": reviewer_identity["display_name"],
                "identity": reviewer_identity,
            }
            if observation.get("model")
            else None,
            "calls": [
                {
                    "duration_ms": number(item.get("duration_ms")),
                    "status": item.get("status"),
                    "response": {
                        key: (item.get("response") or {})[key]
                        for key in ("usage", "timings")
                        if key in (item.get("response") or {})
                    },
                }
                for item in calls
            ],
            "duration_basis": "recorded calls" if "calls" in observation else "legacy evaluation interval",
        },
    }


def experiment_overview(records, lifecycle=None):
    rows = [{**row, "number": index + 1} for index, row in enumerate(records)]
    completed = [row for row in rows if row["status"] == "completed" and row.get("backend") != "demo"]
    measured = [row for row in completed if row["generator_duration_ms"] is not None]
    duration = elapsed((lifecycle or {}).get("started_at"), (lifecycle or {}).get("finished_at"))
    calls = [
        call
        for row in rows
        if row.get("backend") != "demo"
        for call in row.get(
            "generator_calls", [{"response": row["response"], "duration_ms": row["generator_duration_ms"]}]
        )
    ]
    evaluator_calls = [call for row in rows for call in row["evaluator"]["calls"]]
    evaluation_times = [call["duration_ms"] for call in evaluator_calls if call["duration_ms"] is not None]
    documents = {
        (doc["document_id"], doc["version"]) for row in rows for doc in row["enabled_documents"] or []
    }

    def distinct(values):
        result = []
        for value in values:
            if value is not None and value not in result:
                result.append(value)
        return result

    def spread(key):
        values = [row[key] for row in rows if row[key] is not None]
        return {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "total": sum(values) if rows and len(values) == len(rows) else None,
            "measured_runs": len(values),
        }

    def public_row(row):
        return (
            {
                key: value
                for key, value in row.items()
                if key not in {"response", "evaluator", "enabled_documents"}
            }
            if row
            else None
        )

    return {
        "version": "experiment-performance-v1",
        "batch_started_at": (lifecycle or {}).get("started_at"),
        "batch_finished_at": (lifecycle or {}).get("finished_at"),
        "total_duration_ms": duration,
        "batch_duration_basis": "persisted batch start/end"
        if duration is not None
        else "unavailable (batch incomplete or historical boundaries missing)",
        "completed_runs": sum(row["status"] == "completed" for row in rows),
        "failed_runs": sum(row["status"] == "failed" for row in rows),
        "average_generator_ms": mean(row["generator_duration_ms"] for row in measured) if measured else None,
        "duration_measured_runs": len(measured),
        "completed_generator_runs": len(completed),
        "fastest": public_row(min(measured, key=lambda row: row["generator_duration_ms"]))
        if measured
        else None,
        "slowest": public_row(max(measured, key=lambda row: row["generator_duration_ms"]))
        if measured
        else None,
        "generator": {
            "tokens": token_summary(calls),
            **speed(
                [
                    call
                    for row in completed
                    for call in row.get(
                        "generator_calls",
                        [{"response": row["response"], "duration_ms": row["generator_duration_ms"]}],
                    )
                ]
            ),
            "json_format": {
                "checked": sum(row.get("json_format", {}).get("initial_valid") is not None for row in rows),
                "initial_valid": sum(row.get("json_format", {}).get("initial_valid") is True for row in rows),
                "repair_attempts": sum(
                    row.get("json_format", {}).get("repair_status") is not None for row in rows
                ),
                "repaired": sum(
                    row.get("json_format", {}).get("repair_status") == "repaired" for row in rows
                ),
            },
        },
        "configuration": {
            "models": distinct(row["model"] for row in rows),
            "temperatures": distinct(row["temperature"] for row in rows),
            "seeds": distinct(row["seed"] for row in rows),
            "unique_enabled_document_versions": len(documents)
            if rows and all(row["enabled_documents"] is not None for row in rows)
            else None,
            "source_fragments": spread("source_fragments"),
            "source_text_characters": spread("source_text_characters"),
            "source_context_tokens": None,
        },
        "evaluator": {
            "configured": any(row["evaluator"]["configured"] for row in rows),
            "models": distinct(row["evaluator"]["model"] for row in rows),
            "calls": len(evaluator_calls),
            "errors": sum(call["status"] == "failed" for call in evaluator_calls),
            "total_duration_ms": sum(evaluation_times)
            if evaluator_calls and len(evaluation_times) == len(evaluator_calls)
            else None,
            "average_duration_ms": mean(evaluation_times) if evaluation_times else None,
            "duration_measured_calls": len(evaluation_times),
            "tokens": token_summary(evaluator_calls),
            "duration_bases": distinct(
                row["evaluator"]["duration_basis"] for row in rows if row["evaluator"]["calls"]
            ),
        },
        "runs": [public_row(row) for row in rows],
    }
