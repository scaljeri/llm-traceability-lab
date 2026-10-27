"""Echte OpenTelemetry-spans, lokaal opgeslagen en optioneel via OTLP geëxporteerd."""

from contextlib import contextmanager

from opentelemetry import context, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.sampling import ALWAYS_ON

from traceability_lab.config import Settings
from traceability_lab.db import Database, json_text

TRACE_SCHEMA = "lab-spans-v1"  # Eigen attributen; geen claim van GenAI/OpenInference-conformiteit.


class SQLiteExporter(SpanExporter):
    def __init__(self, db: Database):
        self.db = db

    def export(self, spans):
        with self.db.connect() as connection:
            for span in spans:
                connection.execute(
                    "INSERT INTO spans VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        f"{span.context.trace_id:032x}",
                        f"{span.context.span_id:016x}",
                        f"{span.parent.span_id:016x}" if span.parent else None,
                        span.name,
                        span.start_time,
                        span.end_time,
                        span.status.status_code.name,
                        json_text(dict(span.attributes or {})),
                        json_text(
                            [
                                {
                                    "name": event.name,
                                    "time_ns": event.timestamp,
                                    "attributes": dict(event.attributes or {}),
                                }
                                for event in span.events
                            ]
                        ),
                    ),
                )
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


class Tracing:
    def __init__(self, db: Database, settings: Settings):
        # Een eigen provider per app voorkomt globale staat tussen tests of instanties.
        self.provider = TracerProvider(
            sampler=ALWAYS_ON,
            resource=Resource.create(
                {
                    "service.name": "ai-traceability-lab",
                    "service.version": "0.1.0",
                }
            ),
        )
        self.provider.add_span_processor(SimpleSpanProcessor(SQLiteExporter(db)))
        if settings.otlp_endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            self.provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(
                        endpoint=settings.otlp_endpoint,
                        timeout=3,
                    )
                )
            )
        self.tracer = self.provider.get_tracer("traceability_lab", "0.1.0", schema_url=None)

    @contextmanager
    def step(self, name: str, attributes: dict | None = None, *, parent=None):
        # Geen prompts, broninhoud of foutmeldingen in telemetry dupliceren.
        with self.tracer.start_as_current_span(
            name,
            context=parent,
            attributes={"lab.schema": TRACE_SCHEMA, **(attributes or {})},
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                yield span
            except Exception as error:
                span.set_status(trace.StatusCode.ERROR)
                span.add_event("operation.failed", {"error.type": type(error).__name__})
                raise

    def run_context(self, trace_id: str, span_id: str):
        """Latere menselijke annotaties linken aan de oorspronkelijke request-span."""
        span = trace.NonRecordingSpan(
            trace.SpanContext(int(trace_id, 16), int(span_id, 16), False, trace.TraceFlags(1))
        )
        return trace.set_span_in_context(span, context.Context())
