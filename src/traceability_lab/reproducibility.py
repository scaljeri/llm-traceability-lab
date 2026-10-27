"""Versioned canonical configuration fingerprints, never model artifact hashes."""

import hashlib
import json

from traceability_lab.model_identity import recorded_identity


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def behavior(identity: dict) -> dict:
    return {
        "artifact": {
            key: identity.get("artifact", {}).get(key)
            for key in (
                "filename",
                "sha256",
                "size_bytes",
                "revision",
                "architecture",
                "quantization",
                "identity_status",
                "field_sources",
            )
        },
        "runtime": {
            key: identity.get("runtime", {}).get(key)
            for key in (
                "engine",
                "revision",
                "configuration",
                "server_configuration",
                "server_defaults",
                "chat_template",
                "chat_template_sha256",
                "model_path",
                "gpu_offloading",
                "field_sources",
            )
        },
        "routing": {key: identity.get("technical", {}).get(key) for key in ("url", "api_model_id")},
    }


def reproducibility_record(run: dict, automatic: dict) -> dict:
    snapshot = run["snapshot"]
    reviewer = automatic.get("semantic_evaluation") or {}
    generator_identity = recorded_identity(snapshot.get("model"), (snapshot.get("profiles") or {}).get("llm"))
    reviewer_identity = (
        recorded_identity(reviewer.get("model"), reviewer.get("profile")) if reviewer else None
    )
    missing = []
    if not snapshot.get("request_body"):
        missing.append("generator.request_parameters_and_prompts")
    for role, identity in (("generator", generator_identity), ("reviewer", reviewer_identity)):
        if identity is None:
            continue
        for field, value in (
            ("artifact_sha256", identity["artifact"].get("sha256")),
            ("artifact_filename", identity["artifact"].get("filename")),
            ("architecture", identity["artifact"].get("architecture")),
            ("quantization", identity["artifact"].get("quantization")),
            ("engine", identity["runtime"].get("engine")),
            ("engine_revision", identity["runtime"].get("revision")),
            ("chat_template", identity["runtime"].get("chat_template")),
            ("server_defaults", identity["runtime"].get("server_defaults")),
        ):
            if value is None:
                missing.append(f"{role}.{field}")
        if identity["artifact"].get("identity_status") != "verified":
            missing.append(f"{role}.loaded_artifact_verification")
    definition = snapshot.get("experiment") or {}
    inputs = {
        "version": "configuration-fingerprint-v2",
        "generator": {
            "identity": behavior(generator_identity),
            "request": snapshot.get("request_body"),
            "prompt_version": snapshot.get("prompt_version"),
            "context_mode": snapshot.get("context_mode"),
            "embedding": snapshot.get("embedding"),
        },
        "reviewer": {
            "identity": behavior(reviewer_identity) if reviewer_identity else None,
            "configuration": {
                key: value
                for key, value in reviewer.get("configuration", {}).items()
                if key != "llm_profile_id"
            },
            "prompt_version": reviewer.get("prompt_version"),
            "schema_version": reviewer.get("schema_version"),
            "policies": reviewer.get("policies"),
            **({"user_prompts": reviewer["user_prompts"]} if reviewer.get("user_prompts") else {}),
            "output_schemas": reviewer.get("output_schemas"),
        },
        "sources": [
            {key: source.get(key) for key in ("document_id", "version", "sha256", "start", "end")}
            for source in run.get("context", [])
        ],
        "definition_version": definition.get("definition_version"),
        "validation_reference": {
            key: value
            for key, value in (snapshot.get("evaluation_reference") or {}).get("definition", {}).items()
            if key in {"semantic_reference", "validation_json", "answer_json", "output_mode"}
        },
        "evaluation_version": automatic.get("evaluator_version"),
        "implementation_sha256": snapshot.get("source_tree_sha256"),
    }
    return {
        "version": inputs["version"],
        "configuration_sha256": hashlib.sha256(canonical(inputs).encode()).hexdigest(),
        "inputs": inputs,
        "missing_provenance_fields": missing,
        "note": "Identifies recorded configuration, not a model artifact or guaranteed identical output.",
    }
