"""One explicit format-only correction; no validation/reference values enter the request."""

from copy import deepcopy
from time import perf_counter

from traceability_lab.cancellation import RunCancelled
from traceability_lab.db import now
from traceability_lab.experiment_evaluation import parse_answer
from traceability_lab.fact_answers import fact_format_reason, quoted_output, valid_fact_answer

POLICY = {"version": "json-format-repair-v1", "max_attempts": 1}


def matches_structure(template, value):
    if isinstance(template, dict):
        return (
            isinstance(value, dict)
            and template.keys() == value.keys()
            and all(matches_structure(item, value[key]) for key, item in template.items())
        )
    if isinstance(template, list):
        return isinstance(value, list)
    return not isinstance(value, (dict, list))


def format_problem(answer, response, question):
    if response.get("choices", [{}])[0].get("finish_reason") == "length":
        return "The JSON answer was cut short by the output token limit."
    parsed = parse_answer(answer)
    if parsed is None:
        return "Invalid JSON: return exactly one JSON object, without Markdown, duplicate keys or surrounding text."
    if question.output_mode == "facts_and_answer":
        valid = valid_fact_answer(parsed)
        if valid and quoted_output(question.answer_json):
            valid = all(
                "source_ids" in fact or all("quote" in source for source in fact["sources"])
                for fact in parsed["facts"]
            )
        if not valid:
            return fact_format_reason(
                {
                    "status": "completed",
                    "answer": answer,
                    "response": response,
                    "snapshot": {
                        "evaluation_reference": {"definition": {"answer_json": question.answer_json}}
                    },
                }
            )
    elif not matches_structure(question.answer_json, parsed):
        return "The JSON object does not match the required keys and structure in the original instructions."
    return None


def repair_json(settings, question, snapshot, response, chunks, tracing, generate, save_snapshot):
    """Return the effective response; retain the original and every repair artifact."""
    if settings.backend == "demo" or not (
        question.answer_json is not None or question.output_mode == "facts_and_answer"
    ):
        return response
    snapshot["json_repair_policy"] = POLICY.copy()
    original = response["choices"][0]["message"]["content"]
    problem = format_problem(original, response, question)
    snapshot["initial_json_format_valid"] = problem is None
    if problem is None:
        save_snapshot()
        return response
    body = deepcopy(snapshot["request_body"])
    body["messages"] += [
        {"role": "assistant", "content": original},
        {
            "role": "user",
            "content": "Correct only the JSON syntax and required output structure of your previous answer. "
            "Use the original instructions and documents above. Do not answer the question again. "
            "Preserve factual statements, source references, quotations and the meaning of the answer; "
            "do not invent, add or remove factual information. Wrap fields in the required containers if needed. "
            "Return exactly one valid JSON object with no Markdown or surrounding text. "
            "A missing value may be null only where the output contract permits it. "
            "Format checker: " + problem,
        },
    ]
    record = snapshot["json_repair"] = {
        "version": POLICY["version"],
        "status": "running",
        "reason": problem,
        "original_answer": original,
        "original_response": deepcopy(response),
        "request_body": body,
        "response": None,
        "started_at": now(),
    }
    save_snapshot()
    started = perf_counter()
    try:
        with tracing.step("llm.repair_json", {"lab.repair_attempt": 1}):
            corrected = generate(settings, body, chunks)
        record["response"] = corrected
        record["answer"] = corrected["choices"][0]["message"]["content"]
        record["format_error"] = format_problem(record["answer"], corrected, question)
        record["status"] = "repaired" if record["format_error"] is None else "failed"
        return corrected
    except RunCancelled:
        record["status"] = "cancelled"
        raise
    except Exception as error:
        record.update(status="error", error=f"{type(error).__name__}: {error}")
        return response
    finally:
        record.update(finished_at=now(), duration_ms=(perf_counter() - started) * 1000)
        save_snapshot()
