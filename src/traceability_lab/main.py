"""HTTP routes validate input, delegate work and return results."""

import base64
import json
import sqlite3
import sys
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prov.model import ProvDocument

from traceability_lab.config import PACKAGE, Settings
from traceability_lab.db import Database, json_text
from traceability_lab.defined_experiments import (
    delete_definition,
    list_definitions,
    prepare_definition_runs,
    run_definition,
    save_definition,
)
from traceability_lab.documents import protect_sources, version_chunks
from traceability_lab.embeddings import Embedder
from traceability_lab.evaluation import batch_summary, evaluate
from traceability_lab.models import (
    ArtifactVerificationInput,
    DefinedExperimentRun,
    ExperimentDefinition,
    ProfileInput,
    QuestionInput,
)
from traceability_lab.pipeline import preview_context, read_run, read_trace
from traceability_lab.profiles import check_profile, list_profiles, read_profile, save_profile
from traceability_lab.prompts import (
    PromptInput,
    delete_prompt,
    list_prompts,
    save_prompt,
    seed_prompts,
)
from traceability_lab.provenance import export_provenance
from traceability_lab.tracing import Tracing
from traceability_lab.transfers import (
    PortableExperiment,
    export_experiment,
    import_experiment,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    db = Database(settings.database)

    @asynccontextmanager
    async def lifespan(app):
        from traceability_lab.initial_experiments import read_initial_experiments, store_initial_experiments

        initial = read_initial_experiments(settings.experiments_directory) if not db.path.exists() else []
        from traceability_lab.runtime import database_process_lock

        with database_process_lock(db.path):
            db.initialize()
            seed_prompts(db, settings.prompts_directory)
            if initial:
                store_initial_experiments(db, initial)
            protect_sources(db)
            app.state.tracing = Tracing(db, settings)
            app.state.embedder = Embedder(settings)
            from traceability_lab.active_executions import ActiveExecutions

            app.state.executions = ActiveExecutions(db, settings, app.state.embedder, app.state.tracing)
            try:
                yield
            finally:
                app.state.executions.close()
                app.state.tracing.provider.shutdown()

    app = FastAPI(
        title="AI Traceability Lab", version="0.1.0", lifespan=lifespan, docs_url=None, redoc_url=None
    )

    @app.middleware("http")
    async def response_headers(request: Request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        )
        return response

    @app.exception_handler(LookupError)
    async def missing(_request, error):
        return JSONResponse({"detail": str(error)}, status_code=404)

    @app.exception_handler(ValueError)
    async def invalid(_request, error):
        return JSONResponse({"detail": str(error)}, status_code=422)

    @app.exception_handler(sqlite3.IntegrityError)
    async def conflict(_request, _error):
        return JSONResponse(
            {"detail": "This record already exists or violates an integrity constraint."}, status_code=409
        )

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "backend": settings.backend,
            "model": settings.llm_model,
            "model_connection_checked": False,
            "jaeger_url": settings.jaeger_url,
            "notice": "Demo mode · no LLM"
            if settings.backend == "demo"
            else "App configuration · connections checked when a run starts.",
        }

    @app.post("/api/library-files/{direction}")
    def library_files(direction: Literal["import", "export"]):
        from traceability_lab.library_files import export_library_files, import_library_files

        executions = app.state.executions
        with executions.lock:
            if any(job["state"]["status"] in {"running", "cancelling"} for job in executions.jobs.values()):
                return JSONResponse(
                    {"detail": "Stop the active experiment before syncing files."}, status_code=409
                )
            try:
                return (import_library_files if direction == "import" else export_library_files)(db, settings)
            except OSError as error:
                return JSONResponse(
                    {"detail": f"File sync failed: {error.strerror or error}"}, status_code=400
                )

    @app.get("/api/profiles")
    def profiles():
        return list_profiles(db)

    @app.post("/api/profile-models")
    def profile_models(data: ProfileInput):
        return check_profile(data.model_dump())

    @app.delete("/api/profiles/{profile_id}")
    def remove_profile(profile_id: str):
        read_profile(db, profile_id)
        with db.connect() as connection:
            connection.execute("DELETE FROM model_profiles WHERE id=?", (profile_id,))
        return {"deleted": profile_id}

    @app.post("/api/profiles", status_code=201)
    def add_profile(data: ProfileInput):
        return save_profile(db, data)

    @app.put("/api/profiles/{profile_id}")
    def update_profile(profile_id: str, data: ProfileInput):
        return save_profile(db, data, profile_id)

    @app.post("/api/profiles/{profile_id}/check")
    def profile_check(profile_id: str):
        return check_profile(read_profile(db, profile_id))

    @app.post("/api/profiles/{profile_id}/verify-artifact")
    def verify_artifact(profile_id: str, data: ArtifactVerificationInput):
        from traceability_lab.profiles import verify_profile_artifact

        try:
            return verify_profile_artifact(db, profile_id, data.path, data.pid)
        except OSError as error:
            raise ValueError(f"Artifact is not accessible on the application host: {error}") from error

    @app.get("/api/experiment-definitions")
    def definitions():
        return list_definitions(db)

    @app.post("/api/experiment-definitions", status_code=201)
    def create_definition(data: ExperimentDefinition):
        return save_definition(db, data)

    @app.get("/api/experiment-definitions/{identity}/export")
    def experiment_export(identity: str):
        return export_experiment(db, identity)

    @app.post("/api/experiment-definitions/import")
    def experiment_import(data: PortableExperiment):
        return import_experiment(db, data)

    @app.put("/api/experiment-definitions/{identity}")
    def update_definition(identity: str, data: ExperimentDefinition):
        return save_definition(db, data, identity)

    @app.delete("/api/experiment-definitions/{identity}")
    def remove_definition(identity: str):
        return delete_definition(db, identity)

    @app.post("/api/experiment-definitions/run", status_code=201)
    def execute_definition(data: DefinedExperimentRun):
        return run_definition(db, settings, app.state.embedder, app.state.tracing, data)

    @app.post("/api/experiment-definitions/stream")
    def stream_definition(data: DefinedExperimentRun):
        events = prepare_definition_runs(db, settings, app.state.embedder, app.state.tracing, data)
        return StreamingResponse(
            (json_text(event) + "\n" for event in events),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/executions", status_code=202)
    def start_execution(data: DefinedExperimentRun):
        result = app.state.executions.start(data)
        if result is None:
            return JSONResponse(
                {"detail": "An experiment is already running. Reconnect to its progress."}, status_code=409
            )
        return result

    @app.get("/api/executions/active")
    def active_execution():
        return app.state.executions.active()

    @app.get("/api/executions/{identity}")
    def execution_status(identity: str, known_completed: int | None = None):
        return app.state.executions.get(identity, known_completed=known_completed)

    @app.post("/api/executions/{identity}/cancel")
    def cancel_execution(identity: str):
        return app.state.executions.cancel(identity)

    @app.get("/api/batches/{batch_id}")
    def batch(batch_id: str):
        return batch_summary(db, batch_id)

    @app.post("/api/context-preview")
    def context_preview(data: QuestionInput):
        if data.context_mode != "explicit" or data.documents is None:
            raise ValueError("Preview requires an experiment with its own ordered documents.")
        if data.context_ids or data.context_override:
            raise ValueError("Use the experiment context instead of the removed instruction library.")
        return preview_context(db, settings, data)

    @app.get("/api/prompts")
    def prompts_list():
        return list_prompts(db)

    @app.post("/api/prompts", status_code=201)
    def prompts_create(data: PromptInput):
        return save_prompt(db, data)

    @app.put("/api/prompts/{prompt_id}")
    def prompts_update(prompt_id: str, data: PromptInput):
        return save_prompt(db, data, prompt_id)

    @app.delete("/api/prompts/{prompt_id}")
    def prompts_delete(prompt_id: str):
        return delete_prompt(db, prompt_id)

    @app.get("/api/runs")
    def runs():
        return db.rows(
            "SELECT id,created_at,status,question,mode,backend,case_id,batch_id, "
            "json_extract(snapshot,'$.profiles.llm.name') AS llm_name, "
            "COALESCE(json_extract(snapshot,'$.model.profile_id'),json_extract(snapshot,'$.profiles.llm.id')) AS profile_id, "
            "COALESCE(json_extract(snapshot,'$.model.display_name'),json_extract(snapshot,'$.profiles.llm.name'),'Unknown model') AS display_name, "
            "COALESCE(json_extract(snapshot,'$.model.display_name'),json_extract(snapshot,'$.profiles.llm.name'),'Unknown model') AS model_id, "
            "json_extract(snapshot,'$.profiles.llm.hardware') AS hardware, "
            "json_extract(snapshot,'$.profiles.llm.precision') AS precision, "
            "json_extract(snapshot,'$.profiles.llm.enable_thinking') AS enable_thinking, "
            "json_extract(snapshot,'$.model.url') AS model_url, "
            "json_extract(snapshot,'$.corpus_sha256') AS corpus_sha256, "
            "json_extract(snapshot,'$.embedding_key') AS embedding_key, "
            "json_extract(snapshot,'$.prompt_version') AS prompt_version, "
            "json_extract(snapshot,'$.question_configuration') AS generation_configuration, "
            "json_extract(response,'$.usage.completion_tokens') AS output_tokens, "
            "(SELECT (end_ns-start_ns)/1000000.0 FROM spans WHERE spans.trace_id=runs.trace_id "
            "AND name='llm.generate' LIMIT 1) AS llm_call_ms FROM runs "
            "ORDER BY created_at DESC LIMIT 200"
        )

    @app.get("/api/runs/{run_id}")
    def run(run_id: str):
        return read_run(db, run_id)

    @app.get("/api/runs/{run_id}/retrievals")
    def retrievals(run_id: str):
        return read_run(db, run_id)["retrievals"]

    @app.get("/api/runs/{run_id}/context")
    def supplied_context(run_id: str):
        return read_run(db, run_id)["context"]

    @app.get("/api/runs/{run_id}/trace")
    def trace(run_id: str):
        return read_trace(db, read_run(db, run_id)["trace_id"])

    @app.get("/api/runs/{run_id}/provenance")
    def provenance(run_id: str, format: Literal["json", "provn"] = "json"):
        payload = export_provenance(db, run_id)
        if format == "provn":
            document = ProvDocument.deserialize(content=json_text(payload), format="json")
            return PlainTextResponse(
                document.serialize(format="provn"),
                headers={
                    "Content-Disposition": f'attachment; filename="provenance-{read_run(db, run_id)["id"]}.provn"',
                },
            )
        return payload

    @app.get("/api/runs/{run_id}/evaluation")
    def evaluation(run_id: str):
        return evaluate(db, run_id)

    @app.get("/api/runs/{run_id}/report")
    def report(run_id: str):
        stored = read_run(db, run_id)
        versions = [
            db.one("SELECT * FROM source_versions WHERE id=?", (item["id"],))
            for item in stored["snapshot"]["corpus"]
        ]
        reviews = db.rows("SELECT * FROM reviews WHERE run_id=? ORDER BY id", (run_id,))
        chunks = version_chunks(db, [item["id"] for item in versions])
        embeddings = []
        if stored["snapshot"].get("embedding_key"):
            placeholders = ",".join("?" for _ in chunks)
            records = db.rows(
                f"SELECT * FROM embeddings WHERE model_key=? AND chunk_id IN ({placeholders})",
                (stored["snapshot"]["embedding_key"], *(chunk["id"] for chunk in chunks)),
            )
            for item in records:
                # Base64 preserves exact float32 bytes without decimal rounding.
                embeddings.append(
                    {
                        **item,
                        "vector": base64.b64encode(item["vector"]).decode("ascii"),
                        "encoding": "base64-float32",
                        "byte_order": sys.byteorder,
                        "configuration": json.loads(item["configuration"]),
                    }
                )
        from traceability_lab.model_identity import recorded_identity

        assessment = evaluate(db, run_id)
        reviewer = (assessment.get("automatic") or {}).get("semantic_evaluation") or {}
        result = {
            "models": {
                "generator": recorded_identity(
                    stored["snapshot"].get("model"), (stored["snapshot"].get("profiles") or {}).get("llm")
                ),
                "reviewer": recorded_identity(reviewer.get("model"), reviewer.get("profile"))
                if reviewer
                else None,
            },
            "reproducibility": assessment.get("reproducibility"),
            "schema": "report-v1",
            "run": stored,
            "documents": versions,
            "chunks": chunks,
            "embeddings": embeddings,
            "evaluation_reference": stored["snapshot"].get("evaluation_reference"),
            "reviews": reviews,
            "trace": read_trace(db, stored["trace_id"]),
            "provenance": export_provenance(db, run_id),
            "evaluation": assessment,
        }
        return JSONResponse(
            result, headers={"Content-Disposition": f'attachment; filename="run-{stored["id"]}.json"'}
        )

    app.mount("/static", StaticFiles(directory=PACKAGE / "web"), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(PACKAGE / "web/index.html")

    @app.get("/docs", include_in_schema=False)
    def api_documentation():
        # Serve API documentation locally without a Swagger CDN.
        return FileResponse(PACKAGE / "web/docs.html")

    return app


app = create_app()
