"""HTTP-routes zijn dun: valideren, één functie aanroepen, resultaat teruggeven."""

import json
from uuid import uuid4

import httpx

from traceability_lab.claims import add_claim, assess_claim
from traceability_lab.config import Settings
from traceability_lab.contexts import list_contexts, reorder_contexts, save_context
from traceability_lab.db import Database, json_text, now
from traceability_lab.documents import ingest, version_chunks
from traceability_lab.embeddings import Embedder
from traceability_lab.models import (
    AssessmentInput,
    ClaimInput,
    ContextInput,
    ContextOrder,
    DocumentInput,
    DocumentTitleInput,
    EmbeddingRequest,
    QuestionInput,
    ReviewInput,
    RevisionInput,
)
from traceability_lab.pipeline import preview_context, read_run, run_question
from traceability_lab.profiles import read_profile
from traceability_lab.retrieval import ensure_embeddings
from traceability_lab.transfers import (
    PortableDocument,
    export_documents,
    import_documents,
)


def install_legacy_api(app, settings):
    """Test-only adapters for historical record compatibility checks."""
    original_routes = list(app.router.routes)
    db = Database(settings.database)

    @app.get("/api/documents")
    def documents():
        return db.rows(
            "SELECT * FROM source_versions WHERE deleted_at IS NULL ORDER BY document_id, imported_at,id"
        )

    @app.get("/api/documents/export")
    def documents_export():
        return export_documents(db)

    @app.post("/api/documents/import")
    def documents_import(data: list[PortableDocument]):
        return import_documents(db, data)

    @app.put("/api/documents/{document_id}/title")
    def rename_document(document_id: str, data: DocumentTitleInput):
        db.one(
            "SELECT document_id FROM source_versions WHERE document_id=? AND deleted_at IS NULL",
            (document_id,),
        )
        with db.connect() as connection:
            connection.execute(
                "UPDATE document_versions SET display_title=? WHERE document_id=?", (data.title, document_id)
            )
        return {"document_id": document_id, "title": data.title}

    @app.delete("/api/document-versions/{version_id}")
    def delete_version(version_id: str):
        db.one("SELECT id FROM source_versions WHERE id=? AND deleted_at IS NULL", (version_id,))
        with db.connect() as connection:
            connection.execute("UPDATE document_versions SET deleted_at=? WHERE id=?", (now(), version_id))
        return {"id": version_id, "deleted": True}

    @app.delete("/api/documents/{document_id}")
    def delete_document(document_id: str):
        db.one(
            "SELECT document_id FROM source_versions WHERE document_id=? AND deleted_at IS NULL",
            (document_id,),
        )
        with db.connect() as connection:
            connection.execute(
                "UPDATE document_versions SET deleted_at=? WHERE document_id=?", (now(), document_id)
            )
        return {"document_id": document_id, "deleted": True}

    @app.get("/api/documents/{document_id}/versions")
    def document_history(document_id: str):
        db.one("SELECT id FROM source_versions WHERE document_id=?", (document_id,))
        return db.rows(
            "SELECT * FROM source_versions WHERE document_id=? ORDER BY imported_at,id", (document_id,)
        )

    @app.get("/api/contexts")
    def contexts():
        return list_contexts(db)

    @app.post("/api/contexts", status_code=201)
    def add_context(data: ContextInput):
        return save_context(db, data)

    @app.put("/api/contexts/{identity}")
    def edit_context(identity: str, data: ContextInput):
        return save_context(db, data, identity)

    @app.post("/api/contexts/order")
    def order_contexts(data: ContextOrder):
        return reorder_contexts(db, data.ids)

    def index_document(version_id: str, profile_id: str | None) -> dict:
        """Saving sources succeeds independently; failed indexing is explicit and retryable."""
        try:
            configuration = settings
            profile = None
            if profile_id:
                profile = read_profile(db, profile_id)
                if profile["kind"] != "embedding":
                    raise ValueError("Select an embedding profile.")
                changes = {
                    "backend": "local",
                    "embedding_url": profile["url"],
                    "embedding_model": profile["model"],
                    "embedding_dimensions": profile["dimensions"],
                    "embedding_revision": profile["revision"],
                    "embedding_server_revision": profile["server_revision"],
                    "embedding_artifact_sha256": profile["artifact_sha256"],
                    "embedding_precision": profile["precision"],
                    "embedding_hardware": profile["hardware"],
                    "embedding_timeout_seconds": profile["timeout_seconds"],
                    "query_prefix": profile["query_prefix"],
                    "document_prefix": profile["document_prefix"],
                }
                configuration = Settings(**{**settings.model_dump(), **changes})
            embedder = Embedder(configuration)
            chunks = version_chunks(db, [version_id])
            with app.state.tracing.step("documents.index", {"lab.version.id": version_id}):
                ensure_embeddings(db, chunks, embedder)
            return {
                "status": "ready",
                "chunks": len(chunks),
                "model_key": embedder.key(),
                "model": {
                    **embedder.configuration(),
                    "display_name": profile["name"] if profile else "Unknown model",
                    "profile_id": profile_id,
                },
            }
        except (httpx.HTTPError, ValueError, LookupError) as error:
            return {"status": "failed", "error": f"{type(error).__name__}: {error}"}

    @app.post("/api/documents", status_code=201)
    def add_document(data: DocumentInput):
        stored = ingest(db, data)
        return {**stored, "indexing": index_document(stored["id"], data.embedding_profile_id)}

    @app.post("/api/documents/{version_id}/embeddings")
    def index_version(version_id: str, data: EmbeddingRequest):
        db.one("SELECT id FROM source_versions WHERE id=?", (version_id,))
        return index_document(version_id, data.embedding_profile_id)

    @app.put("/api/documents/{version_id}")
    def edit_document(version_id: str, data: RevisionInput):
        original = db.one("SELECT * FROM source_versions WHERE id=?", (version_id,))
        if data.version_number is not None and str(data.version_number) != original["version"]:
            raise ValueError("Editing keeps the version number. Use Add version to create another version.")
        target = data.document_id or original["document_id"]
        stored = ingest(
            db,
            DocumentInput(
                document_id=target,
                version=original["version"],
                title=data.title,
                content=data.content,
                filename=data.filename or original["filename"],
                source=data.filename
                if data.filename and data.filename != original["filename"]
                else original["source"],
                metadata=json.loads(original["metadata"]),
                previous_version_id=original["previous_version_id"]
                if target == original["document_id"]
                else None,
                sequence_number=original["sequence_number"],
                published_on=original["published_on"],
                effective_on=original["effective_on"],
            ),
            replace_id=version_id,
        )
        return {**stored, "indexing": index_document(stored["id"], data.embedding_profile_id)}

    @app.post("/api/documents/{version_id}/revisions", status_code=201)
    def revise_document(version_id: str, data: RevisionInput):
        original = db.one("SELECT * FROM source_versions WHERE id=?", (version_id,))
        target_document_id = data.document_id or original["document_id"]
        history = db.rows(
            "SELECT version,sequence_number FROM source_versions WHERE document_id=?",
            (target_document_id,),
        )
        # Existing versions remain immutable. A save always creates a new revision.
        numeric = [
            int(item["version"])
            for item in history
            if item["version"].isascii() and item["version"].isdigit()
        ]
        label = (
            str(data.version_number)
            if data.version_number is not None
            else (str(max(numeric) + 1) if numeric else "edit-" + uuid4().hex[:12])
        )
        sequence = (
            data.version_number
            if data.version_number is not None
            else max((item["sequence_number"] or 0 for item in history), default=0) + 1
        )
        stored = ingest(
            db,
            DocumentInput(
                document_id=target_document_id,
                version=label,
                title=data.title,
                content=data.content,
                source=data.filename or original["source"],
                filename=data.filename,
                metadata=json.loads(original["metadata"]),
                previous_version_id=version_id if target_document_id == original["document_id"] else None,
                sequence_number=sequence,
            ),
        )
        return {**stored, "indexing": index_document(stored["id"], data.embedding_profile_id)}

    @app.get("/api/documents/{version_id}/chunks")
    def chunks(version_id: str):
        db.one("SELECT id FROM source_versions WHERE id=?", (version_id,))
        return version_chunks(db, [version_id])

    @app.post("/api/context-preview")
    def context_preview(data: QuestionInput):
        return preview_context(db, settings, data)

    @app.post("/api/questions", status_code=201)
    def question(data: QuestionInput):
        run_id = run_question(db, settings, app.state.embedder, app.state.tracing, data)
        from traceability_lab.retention import prune_batches

        prune_batches(db, protect="single:" + run_id)
        return read_run(db, run_id)

    @app.get("/api/runs/{run_id}/claims")
    def claims(run_id: str):
        return read_run(db, run_id)["claims"]

    @app.post("/api/runs/{run_id}/claims", status_code=201)
    def claim(run_id: str, data: ClaimInput):
        return add_claim(db, app.state.tracing, run_id, data)

    @app.post("/api/claims/{claim_id}/assessments", status_code=201)
    def assessment(claim_id: str, data: AssessmentInput):
        return assess_claim(db, app.state.tracing, claim_id, data)

    @app.post("/api/runs/{run_id}/reviews", status_code=201)
    def review(run_id: str, data: ReviewInput):
        completed = read_run(db, run_id)
        if completed["status"] != "completed":
            raise ValueError("Only a completed answer can be reviewed.")
        with db.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO reviews (run_id,assessment,created_at) VALUES (?,?,?)",
                (run_id, json_text(data.model_dump()), now()),
            )
            identity = cursor.lastrowid
        return {"id": identity, **data.model_dump()}

    additions = [route for route in app.router.routes if route not in original_routes]
    app.router.routes[:] = additions + [
        route for route in original_routes if getattr(route, "path", None) != "/api/context-preview"
    ]


def legacy_app(settings):
    """Historical-storage test harness; these routes are absent from the public app."""
    from traceability_lab.main import create_app

    app = create_app(settings)
    install_legacy_api(app, settings)
    return app
