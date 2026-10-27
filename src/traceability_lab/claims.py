"""Handmatige annotatie is zichtbaar mensenwerk, geen automatische ground truth."""

from uuid import uuid4

from traceability_lab.db import Database, now
from traceability_lab.models import AssessmentInput, ClaimInput
from traceability_lab.pipeline import read_run
from traceability_lab.tracing import Tracing


def add_claim(db: Database, tracing: Tracing, run_id: str, data: ClaimInput) -> dict:
    run = read_run(db, run_id)
    answer = run["answer"] or ""
    if run["status"] != "completed" or not 0 <= data.start < data.end <= len(answer):
        raise ValueError("The claim must be an existing span in a completed answer.")
    if not answer[data.start : data.end].strip():
        raise ValueError("A claim cannot be empty.")
    claim_id = uuid4().hex
    with tracing.step(
        "claims.annotate",
        {"lab.run.id": run_id, "lab.method": "manual"},
        parent=tracing.run_context(run["trace_id"], run["snapshot"]["root_span_id"]),
    ):
        with db.connect() as connection:
            connection.execute(
                "INSERT INTO claims VALUES (?,?,?,?,?,?,?)",
                (
                    claim_id,
                    run_id,
                    data.start,
                    data.end,
                    answer[data.start : data.end],
                    data.annotator,
                    now(),
                ),
            )
    return db.one("SELECT * FROM claims WHERE id=?", (claim_id,))


def assess_claim(db: Database, tracing: Tracing, claim_id: str, data: AssessmentInput) -> dict:
    claim = db.one("SELECT * FROM claims WHERE id=?", (claim_id,))
    run = read_run(db, claim["run_id"])
    quote = None
    if data.chunk_id:
        chunk = db.one("SELECT * FROM chunks WHERE id=?", (data.chunk_id,))
        eligible = {version["id"] for version in run["snapshot"]["corpus"]}
        if chunk["version_id"] not in eligible:
            raise ValueError("Evidence must come from the frozen corpus selection of this run.")
        if data.scope == "retrieved" and data.chunk_id not in {r["chunk_id"] for r in run["context"]}:
            raise ValueError(
                "This chunk was not supplied to the model. Select corpus as the assessment scope."
            )
        if not 0 <= data.start < data.end <= len(chunk["text"]):
            raise ValueError("The evidence span is outside the chunk.")
        quote = chunk["text"][data.start : data.end]
        if not quote.strip():
            raise ValueError("An evidence span cannot be empty.")
    with tracing.step(
        "claims.assess",
        {"lab.run.id": run["id"], "lab.method": "manual"},
        parent=tracing.run_context(run["trace_id"], run["snapshot"]["root_span_id"]),
    ):
        with db.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO assessments (claim_id,classification,scope,chunk_id,start,end,quote,annotator,note,created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    claim_id,
                    data.classification,
                    data.scope,
                    data.chunk_id,
                    data.start,
                    data.end,
                    quote,
                    data.annotator,
                    data.note,
                    now(),
                ),
            )
            assessment_id = cursor.lastrowid
    return db.one("SELECT * FROM assessments WHERE id=?", (assessment_id,))
