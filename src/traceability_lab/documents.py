"""Van origineel document naar vaste versies en exact terugvindbare chunks."""

import hashlib
import json
import sqlite3
from contextlib import nullcontext
from uuid import uuid4

from traceability_lab.db import Database, json_text, now
from traceability_lab.models import DocumentInput

CHUNKER = "characters-v1:size=500:no-overlap:python-codepoints"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_text(text: str, size: int = 500) -> list[dict]:
    """Bewust simpel: vaste tekenvensters zonder trimmen of normaliseren.

    Offsets zijn halfopen Python/Unicode-tekenposities, geen UTF-8-byteposities.
    Daardoor geldt altijd: originele_tekst[start:end] == chunktekst.
    """
    if size < 1:
        raise ValueError("Chunk size must be positive.")
    return [
        {
            "position": position,
            "start": start,
            "end": min(start + size, len(text)),
            "text": text[start : start + size],
            "sha256": digest(text[start : start + size]),
        }
        for position, start in enumerate(range(0, len(text), size))
    ]


def ingest(
    db: Database,
    document: DocumentInput,
    *,
    replace_id: str | None = None,
    connection: sqlite3.Connection | None = None,
) -> dict:
    """Bewaar bronrevisies; expliciet bewerken vervangt atomair de actieve kopie."""
    identity = digest(json_text([document.document_id, document.version]))
    values = document.model_dump(mode="json")
    with nullcontext(connection) if connection is not None else db.connect() as connection:
        if replace_id:
            connection.execute("BEGIN IMMEDIATE")
        if (
            replace_id
            and not connection.execute(
                "SELECT id FROM source_versions WHERE id=? AND deleted_at IS NULL", (replace_id,)
            ).fetchone()
        ):
            raise ValueError(
                "This document was changed or removed. Refresh the document list before editing."
            )
        catalog = connection.execute(
            "SELECT * FROM source_versions WHERE document_id=?", (document.document_id,)
        ).fetchone()
        existing = connection.execute(
            "SELECT * FROM source_versions WHERE document_id=? AND version=? AND deleted_at IS NULL",
            (document.document_id, document.version),
        ).fetchone()
        if existing:
            if replace_id and existing["id"] != replace_id:
                raise ValueError("Another document already uses this document ID and version.")
            comparable = {key: existing[key] for key in ("title", "source", "filename", "version", "content")}
            same_content = (
                comparable == {key: values[key] for key in comparable}
                and json.loads(existing["metadata"]) == document.metadata
            )
            same_history = all(
                existing[key] == values[key]
                for key in ("previous_version_id", "sequence_number", "published_on", "effective_on")
            )
            if same_content and same_history:
                return dict(existing)
            if not replace_id:
                raise ValueError(
                    "This document version exists with different content or metadata. Choose a new version."
                )
        if replace_id:
            # One atomic replacement of the working copy. Historical runs keep the old ID.
            connection.execute("UPDATE document_versions SET deleted_at=? WHERE id=?", (now(), replace_id))
        if connection.execute("SELECT id FROM document_versions WHERE id=?", (identity,)).fetchone():
            identity = uuid4().hex
        if document.previous_version_id:
            previous = connection.execute(
                "SELECT * FROM source_versions WHERE id=?", (document.previous_version_id,)
            ).fetchone()
            if not previous or previous["document_id"] != document.document_id:
                raise ValueError("The predecessor must exist and belong to the same document.")
            if (
                document.sequence_number
                and previous["sequence_number"]
                and document.sequence_number <= previous["sequence_number"]
            ):
                raise ValueError("The sequence number must be greater than the predecessor's.")
        connection.execute(
            "INSERT INTO document_versions (id,document_id,version,title,filename,source_uri,body,sha256,created_at,metadata,previous_version_id,sequence_number,published_on,effective_on,display_title) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                identity,
                document.document_id,
                document.version,
                document.title,
                document.filename,
                document.source,
                document.content,
                digest(document.content),
                now(),
                json_text(document.metadata),
                document.previous_version_id,
                document.sequence_number,
                values["published_on"],
                values["effective_on"],
                catalog["document_title"] if catalog else document.title,
            ),
        )
        for chunk in chunk_text(document.content):
            chunk_id = digest(json_text([identity, CHUNKER, chunk["position"], chunk["sha256"]]))
            connection.execute(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?)",
                (
                    chunk_id,
                    identity,
                    chunk["position"],
                    chunk["start"],
                    chunk["end"],
                    chunk["text"],
                    chunk["sha256"],
                    CHUNKER,
                ),
            )
        return dict(connection.execute("SELECT * FROM source_versions WHERE id=?", (identity,)).fetchone())


def protect_sources(db: Database):
    """Ook directe SQL-updates mogen de bronketen niet stil veranderen."""
    with db.connect() as connection:
        for table in (
            "chunks",
            "embeddings",
            "claims",
            "assessments",
            "retrievals",
            "context_items",
            "reviews",
            "spans",
            "run_evaluations",
        ):
            for action in ("UPDATE", "DELETE"):
                connection.execute(
                    f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{action.lower()} "
                    f"BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT, 'immutable research record'); END"
                )
        connection.executescript("""
            CREATE TRIGGER IF NOT EXISTS immutable_document_versions_update
            BEFORE UPDATE OF id,document_id,version,title,filename,source_uri,body,sha256,created_at,
                metadata,previous_version_id,sequence_number,published_on,effective_on ON document_versions
            BEGIN SELECT RAISE(ABORT, 'immutable research record'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_document_versions_delete
            BEFORE DELETE ON document_versions
            BEGIN SELECT RAISE(ABORT, 'immutable research record'); END;

            CREATE TRIGGER IF NOT EXISTS context_insert_running
            BEFORE INSERT ON context_items
            WHEN (SELECT status FROM runs WHERE id=NEW.run_id) != 'running'
            BEGIN SELECT RAISE(ABORT, 'immutable completed context'); END;
            CREATE TRIGGER IF NOT EXISTS completed_runs_immutable
            BEFORE UPDATE ON runs WHEN OLD.status != 'running'
            BEGIN SELECT RAISE(ABORT, 'immutable completed run'); END;
            CREATE TRIGGER IF NOT EXISTS runs_no_delete
            BEFORE DELETE ON runs
            BEGIN SELECT RAISE(ABORT, 'immutable research record'); END;
        """)


def version_chunks(db: Database, version_ids: list[str]) -> list[dict]:
    placeholders = ",".join("?" for _ in version_ids)
    return db.rows(
        f"SELECT c.*, v.document_id, v.version, v.title, v.source, v.sequence_number, v.published_on, v.effective_on, v.previous_version_id FROM chunks c "
        f"JOIN source_versions v ON v.id=c.version_id WHERE c.version_id IN ({placeholders}) "
        "ORDER BY c.version_id, c.position",
        version_ids,
    )
