"""Portable JSON transfers: user document labels/version, never database identities."""

import json
from typing import Literal
from urllib.parse import quote

from pydantic import Field, field_validator, model_validator

from traceability_lab.db import Database
from traceability_lab.documents import digest, ingest, version_chunks
from traceability_lab.models import DocumentInput, ExperimentDefinition, ExperimentDocument, Input


def portable_id(document_id: str, version: str) -> str:
    return document_id + "@" + quote(version, safe="")


class PortableDocument(Input):
    id: str | None = None
    document_id: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    version: str = Field(min_length=1, max_length=80)
    title: str = Field(default="", max_length=200)
    filename: str | None = Field(default=None, max_length=500)
    source: str = Field(default="", max_length=500)
    content: str = Field(max_length=100_000)
    metadata: dict[str, str] = Field(default_factory=dict)
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @field_validator("version", mode="before")
    @classmethod
    def numeric_version(cls, value):
        return str(value) if type(value) is int else value

    @model_validator(mode="after")
    def consistent(self):
        if self.id is not None and self.id != portable_id(self.document_id, self.version):
            raise ValueError("Document id must equal document_id@version (URL-encoded version).")
        if self.sha256 is not None and self.sha256 != digest(self.content):
            raise ValueError("Document content does not match its SHA-256.")
        return self

    def document(self) -> DocumentInput:
        return DocumentInput(**self.model_dump(exclude={"id", "sha256"}))


class PortableExperiment(Input):
    format: Literal["traceability-experiment-v1", "traceability-experiment-v2"]
    experiment: dict
    document_hashes: dict[str, str] = Field(default_factory=dict)
    source_ids: dict[str, list[str]] = Field(default_factory=dict)


def document_record(row: dict) -> dict:
    return {
        "id": portable_id(row["document_id"], row["version"]),
        **{
            key: row[key]
            for key in ("document_id", "version", "title", "filename", "source", "content", "sha256")
        },
        "metadata": json.loads(row["metadata"]),
    }


def export_documents(db: Database) -> list[dict]:
    return [
        document_record(row)
        for row in db.rows(
            "SELECT * FROM source_versions WHERE deleted_at IS NULL ORDER BY document_id,version"
        )
    ]


def import_documents(db: Database, documents: list[PortableDocument]) -> dict:
    """All-or-nothing; identical versions are reusable, conflicts never overwrite."""
    if len(documents) > 1000:
        raise ValueError("Import at most 1,000 document versions at once.")
    ids = [portable_id(item.document_id, item.version) for item in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("The import contains duplicate document IDs and versions.")
    created, unchanged = [], []
    with db.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        for item in documents:
            document = item.document()
            existing = connection.execute(
                "SELECT * FROM source_versions WHERE document_id=? AND version=? AND deleted_at IS NULL",
                (document.document_id, document.version),
            ).fetchone()
            if existing:
                current = document_record(dict(existing))
                desired = {
                    "id": portable_id(document.document_id, document.version),
                    **document.model_dump(
                        exclude={
                            "embedding_profile_id",
                            "previous_version_id",
                            "sequence_number",
                            "published_on",
                            "effective_on",
                        }
                    ),
                    "sha256": digest(document.content),
                }
                if current != desired:
                    raise ValueError(
                        f"Document {current['id']} already exists with different content or metadata. Nothing imported."
                    )
                unchanged.append(current["id"])
            else:
                stored = ingest(db, document, connection=connection)
                created.append(portable_id(stored["document_id"], stored["version"]))
    return {"created": created, "unchanged": unchanged}


def owned_source_identifiers(db: Database, documents: list[ExperimentDocument]) -> dict[str, list[str]]:
    from traceability_lab.owned_documents import owned_sources

    result = {}
    for doc in documents:
        if not doc.content:
            continue
        _, chunks = owned_sources(db, [doc.model_copy(update={"enabled": True})])
        result[portable_id(doc.document_id, doc.version)] = [chunk["id"] for chunk in chunks]
    return result


def export_experiment(db: Database, identity: str) -> dict:
    from traceability_lab.defined_experiments import list_definitions

    saved = next((item for item in list_definitions(db) if item["id"] == identity), None)
    if saved is None:
        raise LookupError("Experiment not found.")
    definition = dict(saved["definition"])
    documents = definition.get("documents")
    if documents is None:
        from traceability_lab.owned_documents import copy_legacy_documents

        documents = copy_legacy_documents(db, definition["version_ids"])
    definition.pop("version_ids", None)
    if definition.get("output_mode") == "facts_and_answer":
        definition.pop("answer_json", None)
        for field in ("system_prompt", "context", "semantic_evaluator"):
            definition.pop(field, None)
    definition["documents"] = [
        {**doc, "id": portable_id(doc["document_id"], doc["version"]), "sha256": digest(doc["content"])}
        for doc in documents
    ]
    return {
        "format": "traceability-experiment-v2",
        "experiment": definition,
        "source_ids": owned_source_identifiers(
            db, [ExperimentDocument.model_validate(doc) for doc in documents]
        ),
    }


def import_experiment(db: Database, data: PortableExperiment) -> dict:
    """Resolve portable labels into a validated draft; saving remains an editor action."""
    definition = dict(data.experiment)
    if data.format == "traceability-experiment-v2":
        documents = definition.get("documents")
        if not isinstance(documents, list):
            raise ValueError("Experiment v2 requires its own documents array.")
        parsed_documents = []
        for item in documents:
            if not isinstance(item, dict):
                raise ValueError("Each experiment document must be an object.")
            enabled = item.get("enabled", True)
            portable = PortableDocument.model_validate(
                {key: value for key, value in item.items() if key != "enabled"}
            )
            parsed_documents.append(
                {
                    **portable.model_dump(exclude={"id", "sha256"}),
                    "filename": portable.filename or "",
                    "enabled": enabled,
                }
            )
        definition["documents"] = parsed_documents
        definition.setdefault("version_ids", [])
        parsed = ExperimentDefinition.model_validate(definition)
        current_sources = owned_source_identifiers(db, parsed.documents)
        warnings = [
            f"Source identifiers changed for {key}. Literal IDs in context and validation are preserved; review them before running."
            for key, value in data.source_ids.items()
            if current_sources.get(key) != value
        ]
        return {"definition": parsed.model_dump(), "warnings": warnings}
    labels = definition.pop("document_ids", None)
    if not isinstance(labels, list) or not labels or not all(isinstance(key, str) for key in labels):
        raise ValueError("An experiment needs an ordered document_ids array of strings.")
    if "version_ids" in definition:
        raise ValueError("Use portable document_ids, not database version_ids.")
    if len(labels) != len(set(labels)):
        raise ValueError("An experiment cannot select the same document version twice.")
    available = {
        portable_id(row["document_id"], row["version"]): row
        for row in db.rows("SELECT * FROM source_versions WHERE deleted_at IS NULL")
    }
    missing = [key for key in labels if key not in available]
    if missing:
        raise ValueError("Import these documents first: " + ", ".join(missing))
    mismatches = [
        key
        for key in labels
        if key in data.document_hashes and data.document_hashes[key] != available[key]["sha256"]
    ]
    if mismatches:
        raise ValueError("Document content hash mismatch: " + ", ".join(mismatches))
    definition["version_ids"] = [available[key]["id"] for key in labels]
    parsed = ExperimentDefinition.model_validate(definition)
    warnings = []
    for key in labels:
        if key in data.source_ids and data.source_ids[key] != [
            c["id"] for c in version_chunks(db, [available[key]["id"]])
        ]:
            warnings.append(
                f"Source identifiers changed for {key}. Literal source IDs in validation or context are preserved; review them before running."
            )
    from traceability_lab.owned_documents import copy_legacy_documents

    owned = parsed.model_dump()
    owned["documents"] = copy_legacy_documents(db, owned["version_ids"])
    owned["version_ids"] = []
    return {"definition": owned, "warnings": warnings}
