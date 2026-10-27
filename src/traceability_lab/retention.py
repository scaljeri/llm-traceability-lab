"""Keep the latest complete executions, removing old runs as one atomic batch."""

from contextlib import contextmanager

from traceability_lab.db import Database, now

BATCH_LIMIT = 100


def prune_batches(db: Database, *, keep: int = BATCH_LIMIT, protect: str | None = None) -> list[str]:
    """Prune logs only; definitions and immutable document bytes stay intact.

    Delete protections are suspended and restored inside one write transaction.
    Other connections cannot write while the protections are suspended; any error
    rolls back both the deletions and trigger changes.
    """
    if keep < 1:
        raise ValueError("Keep at least one execution.")
    with db.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        groups = connection.execute(
            "SELECT COALESCE(r.batch_id,'single:'||r.id) AS execution, MIN(r.created_at) AS started "
            "FROM runs r LEFT JOIN batch_executions b ON b.id=r.batch_id "
            "GROUP BY COALESCE(r.batch_id,'single:'||r.id) "
            "HAVING SUM(r.status='running')=0 AND MAX(b.id IS NOT NULL AND b.finished_at IS NULL)=0 "
            "ORDER BY started DESC,execution DESC"
        ).fetchall()
        ordered = [row["execution"] for row in groups]
        retained = [protect] if protect in ordered else []
        retained.extend(key for key in ordered if key != protect)
        victims = retained[keep:]
        if not victims:
            return []
        connection.execute("CREATE TEMP TABLE retention_runs (id TEXT PRIMARY KEY, trace_id TEXT)")
        for key in victims:
            if key.startswith("single:"):
                connection.execute(
                    "INSERT INTO retention_runs SELECT id,trace_id FROM runs WHERE id=? AND batch_id IS NULL",
                    (key[7:],),
                )
            else:
                connection.execute(
                    "INSERT INTO retention_runs SELECT id,trace_id FROM runs WHERE batch_id=?", (key,)
                )
        tables = (
            "assessments",
            "reviews",
            "claims",
            "run_evaluations",
            "retrievals",
            "context_items",
            "spans",
        )
        trigger_names = [f"immutable_{table}_delete" for table in tables] + ["runs_no_delete"]
        triggers = []
        for name in trigger_names:
            trigger = connection.execute(
                "SELECT name,sql FROM sqlite_master WHERE type='trigger' AND name=?", (name,)
            ).fetchone()
            if trigger:
                triggers.append(dict(trigger))
                connection.execute(f'DROP TRIGGER "{name}"')
        connection.execute(
            "DELETE FROM assessments WHERE claim_id IN (SELECT id FROM claims WHERE run_id IN (SELECT id FROM retention_runs))"
        )
        for table in ("reviews", "claims", "run_evaluations", "retrievals", "context_items"):
            connection.execute(f"DELETE FROM {table} WHERE run_id IN (SELECT id FROM retention_runs)")
        connection.execute(
            "DELETE FROM spans WHERE trace_id IN (SELECT trace_id FROM retention_runs) "
            "AND NOT EXISTS (SELECT 1 FROM runs r WHERE r.trace_id=spans.trace_id AND r.id NOT IN (SELECT id FROM retention_runs))"
        )
        connection.execute("DELETE FROM runs WHERE id IN (SELECT id FROM retention_runs)")
        for key in victims:
            connection.execute("DELETE FROM batch_executions WHERE id=? AND finished_at IS NOT NULL", (key,))
        for trigger in triggers:
            connection.execute(trigger["sql"])
        return victims


@contextmanager
def retained_batch(db: Database, identity: str):
    """Protect the full execution, including gaps between repetitions and evaluation."""
    with db.connect() as connection:
        connection.execute("INSERT INTO batch_executions VALUES (?,?,NULL)", (identity, now()))
    try:
        yield
    finally:
        with db.connect() as connection:
            connection.execute("UPDATE batch_executions SET finished_at=? WHERE id=?", (now(), identity))
        prune_batches(db, protect=identity)
