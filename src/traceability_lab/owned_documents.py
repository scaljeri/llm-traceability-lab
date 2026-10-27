"""Experiment-owned text with immutable, deduplicated provenance snapshots.

Ownership is in the saved definition. Runtime snapshots are never mutable documents
in the shared library; edits in one experiment cannot alter another definition.
"""

import json

from traceability_lab.db import Database, json_text, now
from traceability_lab.documents import CHUNKER, chunk_text, digest, version_chunks
from traceability_lab.models import ExperimentDocument


def owned_sources(db: Database, documents: list[ExperimentDocument], *, persist: bool = False):
    versions, chunks = [], []
    enabled = [doc for doc in documents if doc.enabled]
    keys = [(doc.document_id, doc.version) for doc in documents]
    if len(keys) != len(set(keys)):
        raise ValueError("Document ID and version must be unique within the experiment.")
    if sum(len(doc.content) for doc in enabled) > 100_000:
        raise ValueError("Enabled experiment documents exceed 100,000 characters.")
    for doc in enabled:
        if not doc.content:
            raise ValueError(f"Enabled document {doc.document_id} has no content.")
        values = {
            "document_id": doc.document_id,
            "version": doc.version,
            "filename": doc.filename or f"{doc.document_id}-v{doc.version}.txt",
            "source": doc.source or doc.filename or f"{doc.document_id}-v{doc.version}.txt",
            "title": doc.title or doc.filename or f"{doc.document_id}-v{doc.version}.txt",
            "content": doc.content,
            "metadata": json_text(doc.metadata),
        }
        # Reuse identical immutable historical bytes, preserving existing citation IDs.
        candidates = db.rows(
            "SELECT * FROM source_versions WHERE document_id=? AND version=? ORDER BY imported_at,id",
            (doc.document_id, doc.version),
        )
        match = next(
            (row for row in candidates if all(row[key] == value for key, value in values.items())), None
        )
        if match:
            versions.append(match)
            chunks.extend(version_chunks(db, [match["id"]]))
            continue
        identity = digest(json_text(["experiment-document-v1", values]))
        timestamp = now()
        row = {
            **values,
            "id": identity,
            "sha256": digest(doc.content),
            "created_at": timestamp,
            "imported_at": timestamp,
            "previous_version_id": None,
            "sequence_number": None,
            "published_on": None,
            "effective_on": None,
        }
        fragments = [
            {
                **row,
                **part,
                "version_id": identity,
                "id": digest(json_text([identity, CHUNKER, part["position"], part["sha256"]])),
            }
            for part in chunk_text(doc.content)
        ]
        if persist:
            with db.connect() as connection:
                # Archived provenance snapshots do not become shared editable library entries.
                connection.execute(
                    "INSERT OR IGNORE INTO document_versions (id,document_id,version,filename,source_uri,body,title,display_title,metadata,sha256,created_at,deleted_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        identity,
                        doc.document_id,
                        doc.version,
                        values["filename"],
                        values["source"],
                        doc.content,
                        values["title"],
                        values["title"],
                        json_text(doc.metadata),
                        row["sha256"],
                        timestamp,
                        timestamp,
                    ),
                )
                for part in fragments:
                    connection.execute(
                        "INSERT OR IGNORE INTO chunks VALUES (?,?,?,?,?,?,?,?)",
                        (
                            part["id"],
                            identity,
                            part["position"],
                            part["start"],
                            part["end"],
                            part["text"],
                            part["sha256"],
                            CHUNKER,
                        ),
                    )
        versions.append(row)
        chunks.extend(fragments)
    return versions, chunks


def copy_legacy_documents(db: Database, version_ids: list[str]) -> list[dict]:
    """Copy selected legacy records into independent definition values, including archived text."""
    result = []
    for identity in version_ids:
        row = db.one("SELECT * FROM source_versions WHERE id=?", (identity,))
        result.append(
            ExperimentDocument(
                **{
                    key: row[key]
                    for key in ("document_id", "version", "filename", "title", "source", "content")
                },
                metadata=json.loads(row["metadata"]),
            ).model_dump()
        )
    return result
