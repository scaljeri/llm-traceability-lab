"""Editable supplemental context, copied into each run before execution."""

from uuid import uuid4

from traceability_lab.db import Database, now
from traceability_lab.documents import digest
from traceability_lab.models import ContextInput


def list_contexts(db: Database) -> list[dict]:
    return db.rows("SELECT * FROM saved_contexts ORDER BY position,created_at,id")


def save_context(db: Database, data: ContextInput, identity: str | None = None) -> dict:
    with db.connect() as connection:
        if identity:
            if not connection.execute("SELECT id FROM saved_contexts WHERE id=?", (identity,)).fetchone():
                raise LookupError("Context not found.")
            connection.execute(
                "UPDATE saved_contexts SET title=?,content=?,active=?,updated_at=? WHERE id=?",
                (data.title, data.content, int(data.active), now(), identity),
            )
        else:
            count = connection.execute("SELECT COUNT(*) FROM saved_contexts").fetchone()[0]
            if count >= 100:
                raise ValueError("At most 100 saved contexts are supported.")
            identity = uuid4().hex
            position = connection.execute(
                "SELECT COALESCE(MAX(position),0)+1 FROM saved_contexts"
            ).fetchone()[0]
            timestamp = now()
            connection.execute(
                "INSERT INTO saved_contexts VALUES (?,?,?,?,?,?,?)",
                (identity, data.title, data.content, int(data.active), position, timestamp, timestamp),
            )
    return db.one("SELECT * FROM saved_contexts WHERE id=?", (identity,))


def reorder_contexts(db: Database, ids: list[str]) -> list[dict]:
    with db.connect() as connection:
        existing = {row[0] for row in connection.execute("SELECT id FROM saved_contexts")}
        if len(ids) != len(set(ids)) or set(ids) != existing:
            raise ValueError("Provide every saved context ID exactly once.")
        for position, identity in enumerate(ids, 1):
            connection.execute(
                "UPDATE saved_contexts SET position=?,updated_at=? WHERE id=?", (position, now(), identity)
            )
    return list_contexts(db)


def resolve_contexts(db: Database, ids: list[str] | None, override: str | None = None) -> list[dict]:
    contexts = list_contexts(db)
    if ids is None:
        chosen = [item for item in contexts if item["active"]]
    else:
        by_id = {item["id"]: item for item in contexts}
        if len(ids) != len(set(ids)) or any(identity not in by_id for identity in ids):
            raise ValueError("Context IDs must exist and be unique.")
        chosen = [by_id[identity] for identity in ids]
    if override is not None:
        if not override:
            return []
        timestamp = now()
        return [
            {
                "id": "run-context-" + digest(override),
                "title": "Run-only context",
                "content": override,
                "active": 1,
                "position": 1,
                "created_at": timestamp,
                "updated_at": timestamp,
                "sha256": digest(override),
                "run_only": True,
            }
        ]
    if sum(len(item["content"]) for item in chosen) > 100_000:
        raise ValueError("Active contexts exceed 100,000 characters.")
    return [{**item, "sha256": digest(item["content"])} for item in chosen]
