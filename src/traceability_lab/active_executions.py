"""One local background batch; browsers observe it without owning its lifetime."""

from copy import deepcopy
from threading import Event, Lock, Thread
from uuid import uuid4

from traceability_lab.defined_experiments import prepare_definition_runs


class ActiveExecutions:
    def __init__(self, db, settings, embedder, tracing):
        self.db, self.settings, self.embedder, self.tracing = db, settings, embedder, tracing
        self.lock = Lock()
        self.jobs = {}

    def active(self):
        with self.lock:
            return next(
                (
                    deepcopy(job["state"])
                    for job in self.jobs.values()
                    if job["state"]["status"] in {"running", "cancelling"}
                ),
                None,
            )

    def get(self, identity, *, known_completed: int | None = None):
        with self.lock:
            if identity not in self.jobs:
                raise LookupError("Execution not found. The backend may have restarted.")
            state = self.jobs[identity]["state"]
            if known_completed == state["completed"] and state["status"] in {"running", "cancelling"}:
                return deepcopy({**state, "summary": None})
            return deepcopy(state)

    def start(self, request):
        with self.lock:
            if any(job["state"]["status"] in {"running", "cancelling"} for job in self.jobs.values()):
                return None
            identity, cancel = uuid4().hex, Event()
            events = prepare_definition_runs(
                self.db,
                self.settings,
                self.embedder,
                self.tracing,
                request,
                cancel_event=cancel,
                batch_id=identity,
            )
            # Keep terminal observations bounded, just like the persisted run history.
            if len(self.jobs) >= 100:
                self.jobs.pop(next(iter(self.jobs)))
            job = {
                "cancel": cancel,
                "state": {
                    "batch_id": identity,
                    "saved_id": request.saved_id,
                    "title": request.definition.title,
                    "total": request.execution.repetitions,
                    "completed": 0,
                    "status": "running",
                    "summary": None,
                    "error": None,
                    "execution": request.execution.model_dump(),
                    "reviewer": request.definition.semantic_evaluator.model_dump()
                    if request.definition.semantic_evaluator
                    else None,
                },
            }
            self.jobs[identity] = job
            thread = Thread(target=self._run, args=(job, events), daemon=True)
            job["thread"] = thread
            thread.start()
            return deepcopy(job["state"])

    def cancel(self, identity):
        with self.lock:
            if identity not in self.jobs:
                raise LookupError("Execution not found.")
            job = self.jobs[identity]
            if job["state"]["status"] in {"running", "cancelling"}:
                job["cancel"].set()
                job["state"]["status"] = "cancelling"
            thread = job["thread"]
        # HTTP cancellation normally finishes within one 50 ms poll plus logging.
        thread.join(timeout=0.25)
        return self.get(identity)

    def close(self):
        with self.lock:
            threads = [job["thread"] for job in self.jobs.values()]
            for job in self.jobs.values():
                job["cancel"].set()
        for thread in threads:
            thread.join(timeout=1)

    def _run(self, job, events):
        from traceability_lab.cancellation import cancellation_scope

        with cancellation_scope(job["cancel"]):
            self._consume(job, events)

    def _consume(self, job, events):
        try:
            for event in events:
                with self.lock:
                    state = job["state"]
                    if event["type"] == "progress":
                        state["completed"] = event["completed"]
                        state["summary"] = event.get("summary")
                    elif event["type"] in {"completed", "cancelled", "error"}:
                        state["summary"] = event.get("summary")
                        state["error"] = event.get("message")
                        state["status"] = "failed" if event["type"] == "error" else event["type"]
        except Exception as error:
            with self.lock:
                job["state"].update(status="failed", error=f"{type(error).__name__}: {error}")
