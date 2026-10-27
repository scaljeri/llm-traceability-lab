"""Whole-execution FIFO retention, complete cleanup and transactional protections."""

import sqlite3

import pytest

from traceability_lab.retention import prune_batches, retained_batch


def add_runs(db, batch, index, count=1):
    ids = []
    with db.connect() as c:
        chunk = c.execute("SELECT id FROM chunks LIMIT 1").fetchone()[0]
        for repetition in range(count):
            identity = f"run-{batch}-{repetition}"
            trace = f"trace-{batch}-{repetition}"
            stamp = f"2026-01-01T00:{index // 60:02d}:{index % 60:02d}Z"
            c.execute(
                "INSERT INTO runs(id,created_at,status,trace_id,question,mode,backend,batch_id,snapshot,answer) VALUES (?,?,'running',?,'Question','strict','demo',?,'{}','ok')",
                (identity, stamp, trace, batch),
            )
            c.execute("INSERT INTO context_items VALUES (?,1,?,'explicit')", (identity, chunk))
            c.execute("UPDATE runs SET status='completed',finished_at=? WHERE id=?", (stamp, identity))
            c.execute("INSERT INTO run_evaluations VALUES (?,?,'{}')", (identity, stamp))
            c.execute("INSERT INTO retrievals VALUES (?,1,?,0,'cosine_distance')", (identity, chunk))
            c.execute("INSERT INTO claims VALUES (?, ?,0,2,'ok','test',?)", (identity, identity, stamp))
            c.execute(
                "INSERT INTO assessments(claim_id,classification,scope,annotator,note,created_at) VALUES (?,'supported','retrieved','test','note',?)",
                (identity, stamp),
            )
            c.execute(
                "INSERT INTO reviews(run_id,assessment,created_at) VALUES (?,'{}',?)", (identity, stamp)
            )
            c.execute(
                "INSERT INTO spans VALUES (?, ?,NULL,'llm.generate',0,100,'UNSET','{}','[]')",
                (trace, identity),
            )
            ids.append(identity)
    return ids


def test_default_limit_is_100_whole_batches(client, db):
    for index in range(101):
        add_runs(db, f"batch-{index:03d}", index, count=2)
    source_count = db.one("SELECT count(*) n FROM chunks")["n"]
    removed = prune_batches(db)
    assert removed == ["batch-000"]
    assert db.one("SELECT count(DISTINCT batch_id) n FROM runs")["n"] == 100
    for table in (
        "runs",
        "context_items",
        "retrievals",
        "claims",
        "assessments",
        "reviews",
        "spans",
        "run_evaluations",
    ):
        assert db.one(f"SELECT count(*) n FROM {table}")["n"] == 200
    assert db.one("SELECT count(*) n FROM chunks")["n"] == source_count
    with db.connect() as c:
        assert c.execute("PRAGMA foreign_key_check").fetchall() == []
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with db.connect() as c:
            c.execute("DELETE FROM run_evaluations")
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with db.connect() as c:
            c.execute("DELETE FROM runs")


def test_active_batch_is_protected_between_repetitions(client, db):
    add_runs(db, "active", 0)
    with retained_batch(db, "active"):
        add_runs(db, "second", 1)
        add_runs(db, "third", 2)
        assert prune_batches(db, keep=1) == ["second"]
        assert db.one("SELECT count(*) n FROM runs WHERE batch_id='active'")["n"] == 1
    assert db.one("SELECT finished_at FROM batch_executions WHERE id='active'")["finished_at"]


def test_deletion_failure_rolls_back_all_records_and_trigger_changes(client, db):
    add_runs(db, "old", 0)
    add_runs(db, "new", 1)
    before = db.rows("SELECT * FROM run_evaluations ORDER BY run_id")
    with db.connect() as c:
        c.execute(
            "CREATE TRIGGER test_block_delete BEFORE DELETE ON runs BEGIN SELECT RAISE(ABORT,'test interruption'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="test interruption"):
        prune_batches(db, keep=1)
    assert db.rows("SELECT * FROM run_evaluations ORDER BY run_id") == before
    assert db.one("SELECT count(*) n FROM runs")["n"] == 2
    assert (
        db.one("SELECT count(*) n FROM sqlite_master WHERE type='trigger' AND name='runs_no_delete'")["n"]
        == 1
    )
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with db.connect() as c:
            c.execute("DELETE FROM run_evaluations")


def test_failed_execution_finishes_lifecycle_and_runs_retention(client, db, monkeypatch):
    add_runs(db, "old", 0)
    calls = []
    monkeypatch.setattr("traceability_lab.retention.prune_batches", lambda db, **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError):
        with retained_batch(db, "failure"):
            raise ValueError("Failed execution")
    assert db.one("SELECT finished_at FROM batch_executions WHERE id='failure'")["finished_at"]
    assert calls == [{"protect": "failure"}]
