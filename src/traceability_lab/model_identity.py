"""Frozen display names, declared artifacts and optional trusted runtime attestations."""

import asyncio
import hashlib
import json
from copy import deepcopy

import httpx

from traceability_lab.db import now


def recorded_identity(model: dict | None, profile: dict | None = None) -> dict:
    """Read only saved data; never look up today's profile for an old run."""
    model, profile = model or {}, profile or {}
    if model.get("display_name"):
        return deepcopy(
            {
                key: model.get(key)
                for key in ("profile_id", "display_name", "artifact", "runtime", "technical")
            }
        )
    sha = model.get("artifact_sha256") or profile.get("artifact_sha256")
    return {
        "profile_id": profile.get("id"),
        "display_name": profile.get("name") or "Unknown model",
        "artifact": {
            "filename": None,
            "sha256": sha,
            "architecture": None,
            "quantization": profile.get("precision"),
            "identity_status": "declared" if sha else "unknown",
            "provenance": "historical_snapshot",
        },
        "runtime": {
            "engine": None,
            "revision": model.get("server_revision") or profile.get("server_revision"),
            "configuration": {},
            "server_configuration": None,
            "server_defaults": None,
            "chat_template": None,
        },
        "technical": {
            "api_model_id": model.get("id") or profile.get("model"),
            "url": model.get("url") or profile.get("url"),
        },
    }


def explain_gaps(identity: dict) -> dict:
    identity["artifact"]["unavailable_reasons"] = {
        field: reason
        for field, reason in {
            "sha256": "The API does not establish a complete GGUF hash; register the actual file or a deployment manifest.",
            "size_bytes": "Complete artifact size requires file registration or deployment metadata.",
            "architecture": "Architecture was not established by the available metadata.",
            "quantization": "Quantization was not established by the available metadata.",
            "filename": "No deployment filename or observed model_path is available.",
        }.items()
        if identity["artifact"].get(field) is None
    }
    identity["runtime"]["unavailable_reasons"] = {
        field: "Not observed from the server and not declared in deployment configuration."
        for field in ("engine", "revision", "chat_template", "server_defaults")
        if identity["runtime"].get(field) is None
    }
    return identity


def capture_identity(
    profile: dict | None,
    *,
    api_model_id: str,
    url: str,
    artifact_sha256: str | None = None,
    server_revision: str | None = None,
    capture_default: bool = False,
) -> dict:
    """No model-content reads/hashing; only metadata and cached stat/process checks."""
    profile = profile or {}
    manifest = profile.get("deployment_manifest") or {}
    sha = manifest.get("sha256") or profile.get("artifact_sha256") or artifact_sha256
    identity = {
        "profile_id": profile.get("id"),
        "display_name": profile.get("name") or "Unknown model",
        "artifact": {
            "filename": manifest.get("filename"),
            "sha256": sha.lower() if sha else None,
            "size_bytes": manifest.get("size_bytes"),
            "revision": profile.get("revision"),
            "architecture": manifest.get("architecture"),
            "quantization": manifest.get("quantization") or profile.get("precision"),
            "identity_status": "declared" if sha or manifest.get("filename") else "unknown",
            "provenance": manifest.get("provenance")
            or ("profile_configuration" if profile else "app_configuration"),
        },
        "runtime": {
            "engine": manifest.get("engine"),
            "revision": manifest.get("engine_revision") or profile.get("server_revision") or server_revision,
            "configuration": {
                "hardware": profile.get("hardware"),
                **manifest.get("runtime_configuration", {}),
            },
            "server_configuration": None,
            "server_defaults": None,
            "chat_template": None,
        },
        "technical": {"api_model_id": api_model_id, "url": url},
    }
    enabled = profile.get("capture_server_metadata")
    identity["artifact"]["field_sources"] = {
        key: "deployment_declared" if value is not None else "unknown"
        for key, value in identity["artifact"].items()
        if key in {"filename", "sha256", "size_bytes", "architecture", "quantization"}
    }
    identity["runtime"]["field_sources"] = {
        key: "deployment_declared" if identity["runtime"].get(key) is not None else "unknown"
        for key in ("engine", "revision", "chat_template", "server_defaults", "server_configuration")
    }
    if enabled is False or enabled is None and not capture_default:
        identity["runtime"]["unavailable_reason"] = (
            "Live metadata capture is disabled for this profile/configuration."
        )
        return explain_gaps(identity)
    try:
        from traceability_lab.cancellation import cancellable_get, cancellation_event

        endpoint, params = url.removesuffix("/v1") + "/props", {"model": api_model_id, "autoload": "false"}
        if cancellation_event() is not None:
            response = asyncio.run(cancellable_get(endpoint, params))
        else:
            with httpx.Client(timeout=3, trust_env=False, follow_redirects=False) as client:
                response = client.get(endpoint, params=params)
                response.raise_for_status()
        props = response.json()
        json.dumps(props, allow_nan=False)
        if not isinstance(props, dict):
            raise ValueError("Server properties must be an object.")
        identity["technical"]["server_metadata"] = {"observed_at": now(), "response": props}
        identity["technical"]["server_metadata"]["server_header"] = response.headers.get("server")
        identity["artifact"]["provenance"] += "; server_metadata:/props"
        if response.headers.get("server", "").lower().startswith("llama.cpp"):
            identity["runtime"]["engine"] = "llama.cpp"
            identity["runtime"]["field_sources"]["engine"] = "server_observed"
        identity["runtime"]["model_path"] = props.get("model_path")
        identity["runtime"]["field_sources"]["model_path"] = (
            "server_observed" if props.get("model_path") else "unknown"
        )
        identity["runtime"]["gpu_offloading"] = props.get("n_gpu_layers")
        identity["runtime"]["gpu_offloading_reason"] = (
            None
            if "n_gpu_layers" in props
            else "Not exposed by this server response; record deployment arguments when known."
        )

        identity["runtime"]["server_defaults"] = (props.get("default_generation_settings") or {}).get(
            "params"
        )
        identity["runtime"]["chat_template"] = props.get("chat_template")
        identity["runtime"]["server_configuration"] = {
            "n_ctx": (props.get("default_generation_settings") or {}).get("n_ctx"),
            "total_slots": props.get("total_slots"),
            "chat_template_caps": props.get("chat_template_caps"),
            "modalities": props.get("modalities"),
        }
        if isinstance(props.get("build_info"), str):
            identity["runtime"]["revision"] = props["build_info"]
            identity["runtime"]["field_sources"]["revision"] = "server_observed"
        if isinstance(props.get("model_ftype"), str):
            identity["artifact"]["quantization"] = props["model_ftype"]
            identity["artifact"]["field_sources"]["quantization"] = "server_observed"
        if isinstance(props.get("model_path"), str):
            identity["artifact"]["field_sources"]["filename"] = "server_observed"
            identity["artifact"]["filename"] = props["model_path"].replace("\\", "/").rsplit("/", 1)[-1]
            if identity["artifact"]["identity_status"] == "unknown":
                identity["artifact"]["identity_status"] = "declared"

        for field in ("chat_template", "server_defaults", "server_configuration"):
            identity["runtime"]["field_sources"][field] = (
                "server_observed" if identity["runtime"].get(field) is not None else "unknown"
            )
        identity["runtime"]["chat_template_sha256"] = (
            hashlib.sha256(props["chat_template"].encode()).hexdigest()
            if isinstance(props.get("chat_template"), str)
            else None
        )
        receipt = profile.get("local_artifact_verification")
        if receipt:
            from traceability_lab.artifact_files import recheck_artifact

            check = recheck_artifact(receipt, profile, props)
            identity["technical"]["local_verification"] = check
            if check["status"] == "file_verified":
                identity["artifact"]["field_sources"].update(
                    sha256="local_file_verified", size_bytes="local_file_verified"
                )
            if check["status"] == "loaded_verified":
                identity["artifact"].update(
                    sha256=receipt["sha256"],
                    size_bytes=receipt["size_bytes"],
                    identity_status="verified",
                    provenance="local_hashed_file_and_live_process_binding",
                )
                identity["runtime"]["server_configuration"]["process_arguments"] = check["process"][
                    "inference_arguments"
                ]
                identity["runtime"]["field_sources"]["process_arguments"] = "local_process_observed"
                identity["artifact"]["field_sources"].update(
                    sha256="locally_verified", size_bytes="locally_verified"
                )
            elif check["status"] == "stale":
                identity["technical"]["verification_warning"] = check["reason"]
        # This is an opt-in deployment extension, NOT a stock llama.cpp field.
        attestation = props.get("traceability_artifact") or {}
        loaded_sha = attestation.get("sha256")
        if (
            identity["artifact"]["identity_status"] != "verified"
            and isinstance(loaded_sha, str)
            and len(loaded_sha) == 64
            and all(char in "0123456789abcdefABCDEF" for char in loaded_sha)
            and attestation.get("api_model_id") == api_model_id
            and attestation.get("loaded") is True
            and isinstance(attestation.get("deployment_id"), str)
            and bool(attestation["deployment_id"].strip())
            and not props.get("is_sleeping", False)
        ):
            if sha and sha.lower() != loaded_sha.lower():
                identity["technical"]["declared_sha256_mismatch"] = sha
            identity["artifact"]["sha256"] = loaded_sha.lower()
            identity["artifact"]["size_bytes"] = (
                attestation.get("size_bytes") or identity["artifact"]["size_bytes"]
            )
            trusted = profile.get("trust_server_identity", False)
            identity["artifact"]["identity_status"] = "verified" if trusted else "declared"
            identity["artifact"]["field_sources"]["sha256"] = (
                "trusted_server_attestation" if trusted else "server_declared"
            )
            identity["artifact"]["provenance"] = (
                "trusted_loaded_artifact_attestation" if trusted else "untrusted_server_declaration"
            )
            identity["artifact"]["filename"] = attestation.get("filename") or identity["artifact"]["filename"]
            identity["artifact"]["architecture"] = (
                attestation.get("architecture") or identity["artifact"]["architecture"]
            )
            identity["artifact"]["quantization"] = (
                attestation.get("quantization") or identity["artifact"]["quantization"]
            )
            identity["runtime"]["engine"] = attestation.get("engine") or identity["runtime"]["engine"]
            identity["runtime"]["revision"] = (
                attestation.get("engine_revision") or identity["runtime"]["revision"]
            )
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as error:
        identity["technical"]["metadata_error"] = f"{type(error).__name__}: {error}"
        identity["runtime"]["unavailable_reason"] = identity["technical"]["metadata_error"]
    return explain_gaps(identity)
