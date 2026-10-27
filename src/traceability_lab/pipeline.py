"""Execution stages and immutable input/output snapshots for one model request.

Intermediate results are persisted before external calls. Failed requests retain
source context and exact input. Generation does not establish factual support.
"""

import hashlib
import json
import platform
import sqlite3
from copy import deepcopy
from importlib.metadata import version
from uuid import uuid4

from opentelemetry.context import Context

from traceability_lab.config import PACKAGE, Settings
from traceability_lab.contexts import resolve_contexts
from traceability_lab.db import Database, json_text, now
from traceability_lab.documents import CHUNKER, digest, version_chunks
from traceability_lab.embeddings import Embedder
from traceability_lab.llm import (
    DOCUMENTS_PLACEHOLDER,
    JSON_PLACEHOLDER,
    PROMPT_VERSION,
    QUESTION_PLACEHOLDER,
    build_messages,
    generate,
    request_body,
)
from traceability_lab.model_identity import capture_identity
from traceability_lab.models import QuestionInput
from traceability_lab.profiles import resolve_profiles
from traceability_lab.retrieval import ensure_embeddings, retrieve
from traceability_lab.tracing import TRACE_SCHEMA, Tracing

REQUIRED_SPANS = [
    "rag.request",
    "sources.embed",
    "question.embed",
    "vector.retrieve",
    "prompt.build",
    "llm.generate",
    "run.persist",
]


def source_fingerprint() -> str:
    """Identify exact application files even without a Git commit."""
    files = sorted(
        path
        for path in PACKAGE.rglob("*")
        if path.is_file() and path.suffix in {".py", ".sql", ".js", ".mjs", ".html", ".css", ".json", ".txt"}
    )
    return digest(
        json_text(
            [
                (path.relative_to(PACKAGE).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest())
                for path in files
            ]
        )
    )


def selected_versions(db: Database, requested: list[str] | None) -> list[dict]:
    versions = db.rows("SELECT * FROM source_versions WHERE deleted_at IS NULL ORDER BY imported_at,id")
    if requested is None:
        # Freeze the latest imported version per logical document in the run.
        latest = {row["document_id"]: row for row in versions}
        selected = list(latest.values())
    else:
        by_id = {row["id"]: row for row in versions}
        selected = [by_id[identity] for identity in requested if identity in by_id]
        if len(selected) != len(set(requested)):
            raise ValueError("A selected document version does not exist.")
    if not selected:
        raise ValueError("Import at least one document first.")
    return selected if requested is not None else sorted(selected, key=lambda row: row["id"])


def context_text(question: QuestionInput, chunks: list[dict], supplemental: list[dict]) -> str:
    """Render only model context, excluding the independently editable question."""
    messages = build_messages(
        question.question,
        chunks,
        supplemental_contexts=supplemental,
        prompt_instructions=question.prompt_instructions
        if DOCUMENTS_PLACEHOLDER.search(question.prompt_instructions)
        or question.output_mode == "facts_and_answer"
        else "",
        answer_json=question.answer_json,
        output_mode=question.output_mode,
    )
    if (
        DOCUMENTS_PLACEHOLDER.search(question.prompt_instructions)
        or question.output_mode == "facts_and_answer"
    ):
        return messages[-1]["content"]
    return "\n\n".join(message["content"] for message in messages[1:]).removesuffix(
        "\n\nQUESTION\n" + question.question
    )


def default_system_prompt(question: QuestionInput) -> str:
    """Use the same instruction builder for preview and generation."""
    return build_messages(
        question.question,
        [],
        version_instructions=question.version_instructions,
        prompt_instructions=question.prompt_instructions,
        answer_json=question.answer_json,
        output_mode=question.output_mode,
    )[0]["content"]


def preview_context(db: Database, settings: Settings, question: QuestionInput) -> dict:
    """Select context without generating an answer or persisting a research run."""
    from traceability_lab.prompts import apply_answer_prompt

    question = apply_answer_prompt(db, question)
    settings, _ = resolve_profiles(db, settings, question)
    if question.documents is not None:
        from traceability_lab.owned_documents import owned_sources

        versions, chunks = owned_sources(db, question.documents)
    else:
        versions = selected_versions(db, question.version_ids)
        chunks = version_chunks(db, [row["id"] for row in versions])
    order = {row["id"]: index for index, row in enumerate(versions)}
    chunks.sort(key=lambda chunk: (order[chunk["version_id"]], chunk["position"]))
    if question.context_mode == "retrieval":
        embedder = Embedder(settings)
        ensure_embeddings(db, chunks, embedder)
        vector = embedder.encode(question.question, role="query")
        chunks = retrieve(db, chunks, vector, embedder.key(), question.top_k)
    elif sum(len(chunk["text"]) for chunk in chunks) > 100_000:
        raise ValueError("Explicit context exceeds 100,000 characters. Select fewer versions.")
    supplemental = resolve_contexts(db, question.context_ids, question.context_override)
    return {
        "system_prompt": build_messages(
            question.question,
            [],
            system_prompt=question.system_prompt,
            version_instructions=question.version_instructions,
            prompt_instructions=question.prompt_instructions,
            answer_json=question.answer_json,
            output_mode=question.output_mode,
        )[0]["content"],
        "text": context_text(question, chunks, supplemental),
        "chunk_ids": [c["id"] for c in chunks],
        "documents": [
            {
                "document_id": row["document_id"],
                "version": row["version"],
                "chunk_ids": [c["id"] for c in chunks if c["version_id"] == row["id"]],
            }
            for row in versions
        ],
    }


def run_question(
    db: Database,
    settings: Settings,
    embedder: Embedder,
    tracing: Tracing,
    question: QuestionInput,
    *,
    case: dict | None = None,
    batch_id: str | None = None,
    profiles_snapshot: dict | None = None,
    contexts_snapshot: list[dict] | None = None,
    experiment_snapshot: dict | None = None,
) -> str:
    if experiment_snapshot is not None:
        experiment_snapshot = deepcopy(experiment_snapshot)
    if profiles_snapshot is None:
        settings, profiles_snapshot = resolve_profiles(db, settings, question)
    if profiles_snapshot:
        # Use a per-run client configuration without mutating shared settings.
        embedder = Embedder(settings)
    supplemental = (
        resolve_contexts(db, question.context_ids, question.context_override)
        if contexts_snapshot is None
        else contexts_snapshot
    )
    if question.documents is not None:
        from traceability_lab.owned_documents import owned_sources

        versions, chunks = owned_sources(db, question.documents, persist=True)
    else:
        versions = selected_versions(db, question.version_ids)
        chunks = version_chunks(db, [row["id"] for row in versions])
    # Explicit context preserves request order; chunk order inside a version is fixed.
    order = {row["id"]: index for index, row in enumerate(versions)}
    chunks.sort(key=lambda chunk: (order[chunk["version_id"]], chunk["position"]))
    if question.context_mode == "explicit" and sum(len(chunk["text"]) for chunk in chunks) > 100_000:
        raise ValueError("Explicit context exceeds 100,000 characters. Select fewer versions.")
    prepared = question.llm_context is not None
    context_edited = False
    if prepared:
        if question.context_chunk_ids is None:
            raise ValueError("Prepared context requires the preview chunk IDs.")
        by_id = {chunk["id"]: chunk for chunk in chunks}
        if len(set(question.context_chunk_ids)) != len(question.context_chunk_ids) or any(
            identity not in by_id for identity in question.context_chunk_ids
        ):
            raise ValueError("Preview chunks must belong to the selected document versions.")
        chunks = [by_id[identity] for identity in question.context_chunk_ids]
        context_edited = question.llm_context != context_text(question, chunks, supplemental)
    run_id = uuid4().hex
    manifest = [
        {
            "id": row["id"],
            "document_id": row["document_id"],
            "filename": row["filename"],
            "version": row["version"],
            "sha256": row["sha256"],
            "previous_version_id": row["previous_version_id"],
            "sequence_number": row["sequence_number"],
            "published_on": row["published_on"],
            "effective_on": row["effective_on"],
            "metadata": json.loads(row["metadata"]),
        }
        for row in versions
    ]
    snapshot = {
        "schema": "run-v2",
        "context_edited": context_edited,
        "system_prompt_edited": question.system_prompt is not None
        and question.system_prompt != default_system_prompt(question),
        "context_prepared": prepared,
        "llm_context_sha256": digest(question.llm_context) if prepared else None,
        "profiles": profiles_snapshot,
        "supplemental_contexts": supplemental,
        "question_configuration": question.model_dump(),
        "corpus": manifest,
        "corpus_sha256": digest(json_text(manifest)),
        "chunker": CHUNKER,
        "prompt_version": PROMPT_VERSION
        + ("+facts-output-v2" if question.output_mode == "facts_and_answer" else "")
        + ("+version-order-v1" if question.version_instructions else "")
        + ("+contexts-v1" if supplemental else "")
        + ("+experiment-instructions-v1" if question.prompt_instructions else "")
        + ("+question-slot-v1" if QUESTION_PLACEHOLDER.search(question.prompt_instructions) else "")
        + (
            "+answer-json-slot-v1"
            if question.answer_json is not None and JSON_PLACEHOLDER.search(question.prompt_instructions)
            else ""
        ),
        "context_mode": question.context_mode,
        "trace_schema": TRACE_SCHEMA,
        "telemetry_content_capture": False,
        "required_spans": REQUIRED_SPANS
        if question.context_mode == "retrieval" and not prepared
        else ["rag.request", "context.select", "prompt.build", "llm.generate", "run.persist"],
        "code_revision": settings.code_revision,
        "source_tree_sha256": source_fingerprint(),
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "dependencies": {
            name: version(name) for name in ["fastapi", "sqlite-vec", "httpx", "prov", "opentelemetry-sdk"]
        },
        "model": {
            **capture_identity(
                (profiles_snapshot or {}).get("llm"),
                api_model_id=settings.llm_model,
                url=settings.llm_url,
                artifact_sha256=settings.llm_artifact_sha256,
                server_revision=settings.llm_server_revision,
                capture_default=settings.runtime_metadata_default and settings.backend == "local",
            ),
            "id": settings.llm_model if settings.backend == "local" else "demo-extractive-v1",
            "artifact_sha256": settings.llm_artifact_sha256,
            "server_revision": settings.llm_server_revision,
            "url": settings.llm_url,
            "artifact_identity_status": "operator_declared" if settings.llm_artifact_sha256 else "unknown",
        },
        "case": case,
        "experiment": experiment_snapshot,
        "otlp_endpoint": settings.otlp_endpoint,
    }
    reviewer = (experiment_snapshot or {}).get("semantic_evaluator_snapshot")
    if reviewer:
        profile = reviewer["profile"]
        reviewer["model"] = capture_identity(
            profile,
            api_model_id=profile["model"],
            url=profile["url"],
            capture_default=settings.runtime_metadata_default,
        )
    snapshot["model"].update(
        artifact_sha256=snapshot["model"]["artifact"]["sha256"],
        server_revision=snapshot["model"]["runtime"]["revision"],
        artifact_identity_status=snapshot["model"]["artifact"]["identity_status"],
    )
    if experiment_snapshot is not None:
        snapshot["evaluation_reference"] = experiment_snapshot["ground_truth"]
    # Each request has its own root trace, including repetitions in a batch.
    with tracing.step(
        "rag.request", {"lab.run.id": run_id, "lab.backend": settings.backend}, parent=Context()
    ) as root:
        trace_id = f"{root.get_span_context().trace_id:032x}"
        snapshot["root_span_id"] = f"{root.get_span_context().span_id:016x}"
        with db.connect() as connection:
            connection.execute(
                "INSERT INTO runs (id,created_at,status,trace_id,question,mode,backend,case_id,batch_id,snapshot) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    now(),
                    "running",
                    trace_id,
                    question.question,
                    question.mode,
                    settings.backend,
                    case["id"] if case else None,
                    batch_id,
                    json_text(snapshot),
                ),
            )

        def save_snapshot():
            with db.connect() as connection:
                connection.execute("UPDATE runs SET snapshot=? WHERE id=?", (json_text(snapshot), run_id))

        try:
            if question.context_mode == "explicit" or prepared:
                with tracing.step(
                    "context.select", {"lab.versions": len(versions), "lab.chunks": len(chunks)}
                ):
                    results = [
                        {**chunk, "rank": index, "origin": "explicit"}
                        for index, chunk in enumerate(chunks, 1)
                    ]
                    if context_edited:
                        results = []
            else:
                # 1. Reuse embeddings keyed by model configuration and chunk identity.
                with tracing.step("sources.embed", {"lab.chunks": len(chunks)}):
                    snapshot["embedding"] = embedder.configuration()
                    snapshot["embedding_key"] = embedder.key()
                    ensure_embeddings(db, chunks, embedder)
                save_snapshot()

                # 2. Persist the query vector for inspection and replay.
                with tracing.step("question.embed"):
                    query_vector = embedder.encode(question.question, role="query")
                    snapshot["query_vector"] = query_vector
                    snapshot["embedding_observations"] = list(embedder.observations)
                save_snapshot()

                # 3. Cosine distance is a distance, not a probability.
                with tracing.step("vector.retrieve", {"lab.top_k": question.top_k}):
                    results = retrieve(db, chunks, query_vector, embedder.key(), question.top_k)
                    with db.connect() as connection:
                        for result in results:
                            connection.execute(
                                "INSERT INTO retrievals VALUES (?,?,?,?,?)",
                                (run_id, result["rank"], result["id"], result["distance"], result["metric"]),
                            )

            # Store only chunks actually supplied, independently of retrieval scores.
            with db.connect() as connection:
                for result in results:
                    connection.execute(
                        "INSERT INTO context_items VALUES (?,?,?,?)",
                        (run_id, result["rank"], result["id"], result.get("origin", "retrieved")),
                    )
            snapshot["context_chunk_ids"] = [result["id"] for result in results]
            save_snapshot()

            # 4. Persist the exact messages sent to the model server.
            with tracing.step("prompt.build"):
                messages = build_messages(
                    question.question,
                    results,
                    version_instructions=question.version_instructions,
                    prompt_instructions=question.prompt_instructions,
                    answer_json=question.answer_json,
                    output_mode=question.output_mode,
                    supplemental_contexts=supplemental,
                    llm_context=question.llm_context,
                    system_prompt=question.system_prompt,
                )
                snapshot["request_body"] = request_body(settings, question, messages)
                with db.connect() as connection:
                    connection.execute(
                        "UPDATE runs SET messages=?,snapshot=? WHERE id=?",
                        (json_text(messages), json_text(snapshot), run_id),
                    )

            # 5. Call the selected model server, except in explicit demo mode.
            with tracing.step(
                "llm.generate",
                {
                    "lab.model": snapshot["model"]["display_name"],
                    "lab.model.profile_id": snapshot["model"]["profile_id"] or "unknown",
                },
            ):
                response = generate(settings, snapshot["request_body"], results)
                answer = response["choices"][0]["message"]["content"]

            from traceability_lab.json_repair import repair_json

            # Store the original before a separately recorded correction request.
            with db.connect() as connection:
                connection.execute(
                    "UPDATE runs SET answer=?,response=? WHERE id=?", (answer, json_text(response), run_id)
                )
            response = repair_json(
                settings, question, snapshot, response, results, tracing, generate, save_snapshot
            )
            answer = response["choices"][0]["message"]["content"]

            # 6. Finish generation; evaluation runs separately.
            with tracing.step("run.persist"):
                with db.connect() as connection:
                    connection.execute(
                        "UPDATE runs SET status='completed',finished_at=?,answer=?,response=? WHERE id=?",
                        (now(), answer, json_text(response), run_id),
                    )
        except Exception as error:
            # Do not retry transport errors; preserve the failed attempt as its own run.
            from traceability_lab.cancellation import RunCancelled

            if isinstance(error, RunCancelled):
                snapshot["cancelled"] = True
            root.set_attribute("lab.failure_type", type(error).__name__)
            snapshot["embedding_observations"] = list(embedder.observations)
            from opentelemetry.trace import StatusCode

            root.set_status(StatusCode.ERROR)
            with db.connect() as connection:
                connection.execute(
                    "UPDATE runs SET status='failed',finished_at=?,error=?,snapshot=? WHERE id=?",
                    (now(), f"{type(error).__name__}: {error}", json_text(snapshot), run_id),
                )
    return run_id


def read_run(db: Database, run_id: str, *, include_annotations: bool = True) -> dict:
    run = db.one("SELECT * FROM runs WHERE id=?", (run_id,))
    for field in ("snapshot", "messages", "response"):
        run[field] = json.loads(run[field]) if run[field] is not None else None
    run["retrievals"] = db.rows(
        "SELECT r.*, c.text,c.start,c.end,c.sha256,c.version_id,v.title,v.version,v.document_id,v.filename,v.source "
        "FROM retrievals r JOIN chunks c ON r.chunk_id=c.id "
        "JOIN source_versions v ON c.version_id=v.id WHERE r.run_id=? ORDER BY r.rank",
        (run_id,),
    )
    run["context"] = db.rows(
        "SELECT x.*, x.position AS rank, c.position AS chunk_position, c.text,c.start,c.end,c.sha256,c.version_id,"
        "v.title,v.version,v.document_id,v.filename,v.source,v.sequence_number,v.published_on,v.effective_on,"
        "r.distance,r.metric FROM context_items x JOIN chunks c ON x.chunk_id=c.id "
        "JOIN source_versions v ON c.version_id=v.id "
        "LEFT JOIN retrievals r ON r.run_id=x.run_id AND r.chunk_id=x.chunk_id "
        "WHERE x.run_id=? ORDER BY x.position",
        (run_id,),
    )
    # Old runs predate the separate context table; their prompt used retrieval order.
    if "context_mode" not in run["snapshot"]:
        run["context"] = [{**item, "origin": "retrieved"} for item in run["retrievals"]]
    # Legacy annotations remain readable, but automated evaluation never loads them.
    if include_annotations:
        run["claims"] = db.rows("SELECT * FROM claims WHERE run_id=? ORDER BY start, end", (run_id,))
        for claim in run["claims"]:
            claim["assessments"] = db.rows(
                "SELECT * FROM assessments WHERE claim_id=? ORDER BY id", (claim["id"],)
            )
    return run


def read_trace(db: Database, trace_id: str) -> list[dict]:
    spans = db.rows("SELECT * FROM spans WHERE trace_id=? ORDER BY start_ns, span_id", (trace_id,))
    for span in spans:
        span["attributes"] = json.loads(span["attributes"])
        span["events"] = json.loads(span["events"])
        span["duration_ms"] = (span["end_ns"] - span["start_ns"]) / 1_000_000
    return spans
