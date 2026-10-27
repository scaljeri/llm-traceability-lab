"""Conservative, versioned checks of observable answers, never internal reasoning.

Only the documented JSON response contract is parsed. Free prose, extra claims,
ambiguous JSON and truncated output require human review. Ground truth is input
to this evaluator only, never to the prompt builder or generating model.
"""

import json
import math
import re

from traceability_lab.documents import digest

EVALUATOR_VERSION = "scenario-evaluation-v2"


def parse_answer(answer: str | None) -> dict | None:
    """Reject duplicate keys and anything outside one JSON object."""

    def unique(pairs: list) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate answer key")
            result[key] = value
        return result

    def invalid_constant(value: str):
        raise ValueError(f"Invalid JSON constant: {value}")

    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("JSON number exceeds the finite float range")
        return number

    text = (answer or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if fenced:
        text = fenced[1]
    try:
        result = json.loads(
            text, object_pairs_hook=unique, parse_constant=invalid_constant, parse_float=finite_float
        )
    except ValueError, TypeError, RecursionError:
        return None
    return result if isinstance(result, dict) else None


def automatic_evaluation(run: dict, bodies: dict[str, str]) -> dict:
    """Return independent applicability and results for a frozen reference."""
    snapshot = run["snapshot"]
    reference = snapshot.get("evaluation_reference")
    experiment = snapshot.get("experiment") or {}
    config = snapshot["question_configuration"]
    definition = experiment.get("definition", {})
    context = run["context"]
    sources = {item["chunk_id"]: item for item in context}
    supplied = {(item["document_id"], item["version"]) for item in context}
    parsed = parse_answer(run.get("answer"))
    criteria = []

    def add(key: str, state: str, reason: str, result: str = "requires_review", **details) -> None:
        criteria.append(
            {
                "id": key,
                "applicability": state,
                "reason": reason,
                "result": result if state == "applicable" else "not_evaluated",
                "method": "deterministic",
                **details,
            }
        )

    if not reference:
        return {
            "evaluator_version": EVALUATOR_VERSION,
            "criteria": [],
            "reason": "No frozen ground truth was recorded; historical expectations are not inferred.",
        }

    def applicability(key: str) -> tuple[str, str]:
        if reference.get("definition_version") != experiment.get("definition_version"):
            return "undetermined", "Reference and experiment definition versions differ."
        if snapshot.get("system_prompt_edited"):
            return "undetermined", "System instructions were edited; original expectations require review."
        if snapshot.get("context_edited"):
            return (
                "undetermined",
                "The model context was edited; original source expectations require review.",
            )
        if (
            key == "retrieval_evidence"
            and config["context_mode"] == "retrieval"
            and snapshot.get("context_prepared")
        ):
            return "undetermined", "Context was frozen from a preview; no run-time retrieval was performed."
        if key == "unsupported_claims":
            return "applicable", "Claim support can be reviewed against the exact supplied context."
        if key == "retrieval_evidence" and config["context_mode"] != "retrieval":
            return "not_applicable", "Controlled context performs no retrieval."
        if run["question"] != definition.get("question"):
            return (
                "undetermined",
                "The LLM question differs from the reference scenario; criterion applicability requires review.",
            )
        if key == "retrieval_evidence":
            if config["context_mode"] != "retrieval":
                return "not_applicable", "Controlled context performs no retrieval."
            required = {(item["document_id"], item["version"]) for item in reference["required_evidence"]}
            selected = {(item["document_id"], item["version"]) for item in snapshot["corpus"]}
            if not required <= selected:
                return "not_applicable", "The required evidence version is outside the selected corpus."
            return "applicable", "Reference evidence is in the configured retrieval corpus."
        needed = (
            [reference["supporting_source"]]
            if key != "version_relation"
            else [reference["older_source"], reference["newer_source"]]
        )
        if not {tuple(pair) for pair in needed} <= supplied:
            if run["status"] != "completed":
                return "undetermined", "The run failed before the required model context was confirmed."
            return "not_applicable", "Required document versions are absent from the model context."
        approved = {
            (item["document_id"], item["version"]): item["body_strip_sha256"]
            for item in reference["approved_sources"]
        }
        for doc in snapshot["corpus"]:
            pair = (doc["document_id"], doc["version"])
            if approved.get(pair) != digest(bodies[doc["id"]].strip()):
                return "undetermined", "Additional or changed source content has no approved expectation."
        if (
            config.get("prompt_instructions", "") != definition.get("prompt_instructions", "")
            or config.get("version_instructions", False) != definition.get("version_instructions", False)
            or snapshot.get("supplemental_contexts")
        ):
            return (
                "undetermined",
                "Changed or additional instructions require review of criterion applicability.",
            )
        return (
            "applicable",
            "Required sources and reference instructions are preserved; source order may vary.",
        )

    def source_for(value) -> dict | None:
        if not isinstance(value, str):
            return None
        return sources.get(value.removeprefix("[").removesuffix("]"))

    def pair(source: dict | None) -> list | None:
        return [source["document_id"], source["version"]] if source else None

    for key in reference.get("criteria", []):
        state, reason = applicability(key)
        if state != "applicable":
            add(key, state, reason)
            continue
        if run["backend"] == "demo" or run["status"] != "completed":
            add(key, state, reason, note="No completed real-model answer; demo output is not model behavior.")
            continue
        if key == "retrieval_evidence":
            required = reference["required_evidence"]
            found = sum(
                any(
                    item["document_id"] == needed["document_id"]
                    and item["version"] == needed["version"]
                    and needed["quote"] in item["text"]
                    for item in run["retrievals"]
                )
                for needed in required
            )
            add(
                key,
                state,
                reason,
                "pass" if found == len(required) and required else "fail",
                matched=found,
                total=len(required),
                scope="Retrieval evidence coverage, not answer correctness",
            )
            continue
        if key == "unsupported_claims":
            add(
                key,
                state,
                reason,
                note="Complete factual claim coverage and entailment require human review.",
                method="human_review_required",
            )
            continue
        if not parsed or (run.get("response") or {}).get("choices", [{}])[0].get("finish_reason") == "length":
            add(key, state, reason, note="No complete unambiguous JSON answer; inspect the raw response.")
            continue
        if key == "budget":
            value = parsed.get("budget_eur")
            if (
                type(value) not in (int, float)
                or (type(value) is float and not math.isfinite(value))
                or value < 0
            ):
                add(key, state, reason, note="budget_eur must be a nonnegative JSON number in euros.")
            else:
                add(
                    key,
                    state,
                    reason,
                    "pass" if value == reference["budget_eur"] else "fail",
                    observed=value,
                    expected=reference["budget_eur"],
                    scope="Declared budget field only",
                )
        elif key == "supporting_version":
            value = parsed.get("supporting_source_id")
            if value is None or not isinstance(value, str):
                add(key, state, reason, note="No structured supporting source citation to inspect.")
            else:
                source = source_for(value)
                valid = pair(source) == reference["supporting_source"] and any(
                    item["quote"] in source["text"] for item in reference["required_evidence"]
                )
                add(
                    key,
                    state,
                    reason,
                    "pass" if valid else "fail",
                    observed=value,
                    scope="Citation in supporting_source_id only; other citations require review",
                )
        elif key == "version_relation":
            relation = parsed.get("version_relation")
            if not isinstance(relation, dict) or not all(
                isinstance(relation.get(field), str) for field in ("older_source_id", "newer_source_id")
            ):
                add(key, state, reason, note="No structured older/newer relation to inspect.")
            else:
                valid = (
                    pair(source_for(relation["older_source_id"])) == reference["older_source"]
                    and pair(source_for(relation["newer_source_id"])) == reference["newer_source"]
                )
                add(
                    key,
                    state,
                    reason,
                    "pass" if valid else "fail",
                    observed=relation,
                    scope="Explicitly reported relationship, not observation of internal reasoning",
                )
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "criteria": criteria,
        "reference_sha256": digest(json.dumps(reference, ensure_ascii=False, sort_keys=True)),
    }
