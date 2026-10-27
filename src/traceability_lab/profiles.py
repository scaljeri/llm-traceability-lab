"""Profielen bewaren, controleren en per run naar vaste instellingen vertalen.

Een profiel mag later veranderen. Een bestaande run gebruikt zijn eigen snapshot
en verandert daardoor nooit mee. Endpointcontroles genereren geen antwoord.
"""

import json
from uuid import uuid4

import httpx

from traceability_lab.config import Settings
from traceability_lab.db import Database, json_text, now
from traceability_lab.models import ProfileInput, QuestionInput


def read_profile(db: Database, profile_id: str) -> dict:
    row = db.one("SELECT * FROM model_profiles WHERE id=?", (profile_id,))
    # Nieuwe optionele metadatavelden krijgen ook bij oudere opgeslagen profielen
    # hun defaults; oude profielrecords hoeven daarvoor niet te worden herschreven.
    raw = json.loads(row["configuration"])
    receipt = raw.pop("_artifact_verification", None)
    configuration = ProfileInput.model_validate(raw).model_dump()
    return {
        "id": row["id"],
        **configuration,
        **({"local_artifact_verification": receipt} if receipt else {}),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def list_profiles(db: Database) -> list[dict]:
    return [
        read_profile(db, row["id"]) for row in db.rows("SELECT id FROM model_profiles ORDER BY created_at,id")
    ]


def save_profile(db: Database, data: ProfileInput, profile_id: str | None = None) -> dict:
    if profile_id:
        existing = read_profile(db, profile_id)
        if existing["kind"] != data.kind:
            raise ValueError("The type of an existing profile cannot change.")
        configuration = data.model_dump()
        receipt = existing.get("local_artifact_verification")
        manifest = configuration.get("deployment_manifest") or {}
        if (
            receipt
            and data.url == existing["url"]
            and data.model == existing["model"]
            and manifest.get("sha256") == receipt["sha256"]
            and manifest.get("filename") == receipt["filename"]
        ):
            configuration["_artifact_verification"] = receipt
        with db.connect() as connection:
            connection.execute(
                "UPDATE model_profiles SET configuration=?,updated_at=? WHERE id=?",
                (json_text(configuration), now(), profile_id),
            )
    else:
        profile_id = uuid4().hex
        timestamp = now()
        with db.connect() as connection:
            connection.execute(
                "INSERT INTO model_profiles VALUES (?,?,?,?)",
                (profile_id, json_text(data.model_dump()), timestamp, timestamp),
            )
    return read_profile(db, profile_id)


def resolve_profiles(db: Database, settings: Settings, question: QuestionInput) -> tuple[Settings, dict]:
    """Geen selectie behoudt de appstandaarden/oefenstand; een selectie vereist beide rollen."""
    if not question.llm_profile_id and not question.embedding_profile_id:
        return settings, {}
    if not question.llm_profile_id or (
        question.context_mode == "retrieval" and not question.embedding_profile_id
    ):
        raise ValueError("Select an LLM profile and, for retrieval, an embedding profile.")
    llm = read_profile(db, question.llm_profile_id)
    if llm["kind"] != "llm":
        raise ValueError("The selected LLM profile has an incorrect type.")
    changes = {
        "backend": "local",
        "llm_url": llm["url"],
        "llm_model": llm["model"],
        "llm_artifact_sha256": llm["artifact_sha256"],
        "llm_server_revision": llm["server_revision"],
        "llm_enable_thinking": llm["enable_thinking"],
        "timeout_seconds": llm["timeout_seconds"],
    }
    profiles = {"llm": llm}
    if question.context_mode == "retrieval":
        embedding = read_profile(db, question.embedding_profile_id)
        if embedding["kind"] != "embedding":
            raise ValueError("The selected embedding profile has an incorrect type.")
        profiles["embedding"] = embedding
        changes.update(
            {
                "embedding_url": embedding["url"],
                "embedding_model": embedding["model"],
                "embedding_dimensions": embedding["dimensions"],
                "embedding_revision": embedding["revision"],
                "embedding_server_revision": embedding["server_revision"],
                "embedding_artifact_sha256": embedding["artifact_sha256"],
                "embedding_precision": embedding["precision"],
                "embedding_hardware": embedding["hardware"],
                "embedding_timeout_seconds": embedding["timeout_seconds"],
                "query_prefix": embedding["query_prefix"],
                "document_prefix": embedding["document_prefix"],
            }
        )
    return Settings(**{**settings.model_dump(), **changes}), profiles


def check_profile(profile: dict) -> dict:
    """Alleen /models lezen. Bereikbaar betekent nog geen geslaagde inference."""
    try:
        with httpx.Client(timeout=5, trust_env=False, follow_redirects=False) as client:
            response = client.get(profile["url"] + "/models")
            response.raise_for_status()
            models = response.json()["data"]
        identities = [
            item["id"] for item in models if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]
        return {
            "reachable": True,
            "model_available": profile["model"] in identities,
            "models": identities,
            "inference_checked": False,
        }
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
        return {
            "reachable": False,
            "model_available": False,
            "models": [],
            "inference_checked": False,
            "error": f"{type(error).__name__}: {error}",
        }


def verify_profile_artifact(db: Database, profile_id: str, path: str, pid: int | None) -> dict:
    """Only this explicit workflow creates a trusted local hash receipt."""
    from pathlib import Path

    from traceability_lab.artifact_files import file_identity, hash_artifact, process_binding
    from traceability_lab.models import DeploymentManifest

    profile = read_profile(db, profile_id)
    receipt = hash_artifact(path)
    receipt.update(url=profile["url"], api_model_id=profile["model"])
    if pid is not None:
        try:
            receipt["process"] = process_binding(receipt["path"], pid, profile["url"])
        except (OSError, ValueError, IndexError) as error:
            receipt["association_error"] = str(error)
    if file_identity(Path(receipt["path"]).stat()) != receipt["file_stat"]:
        raise ValueError("Artifact changed during registration; verify again.")
    manifest = {
        **(profile.get("deployment_manifest") or {}),
        "filename": receipt["filename"],
        "sha256": receipt["sha256"],
        "size_bytes": receipt["size_bytes"],
        "provenance": "explicit local complete-GGUF SHA-256 registration",
        "verification": {
            "method": "complete_file_sha256",
            "verified_at": receipt["verified_at"],
            "sha256": receipt["sha256"],
        },
    }
    configuration = {key: value for key, value in profile.items() if key in ProfileInput.model_fields}
    configuration["deployment_manifest"] = DeploymentManifest.model_validate(manifest).model_dump()
    configuration["_artifact_verification"] = receipt
    with db.connect() as connection:
        connection.execute(
            "UPDATE model_profiles SET configuration=?,updated_at=? WHERE id=?",
            (json_text(configuration), now(), profile_id),
        )
    return read_profile(db, profile_id)
