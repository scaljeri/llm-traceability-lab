"""Alle schrijfverzoeken worden hier gevalideerd voordat ze de database raken."""

import json
from datetime import date
from typing import Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from traceability_lab.config import service_url


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    @field_validator("output_mode", mode="before", check_fields=False)
    @classmethod
    def legacy_output_mode(cls, value):
        return "facts_and_answer" if value == "facts_and_conclusion" else value


class ContextInput(Input):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=20_000)
    active: bool = True

    @model_validator(mode="after")
    def meaningful(self):
        if not self.title.strip() or not self.content.strip():
            raise ValueError("Context title and content cannot be blank.")
        return self


class ContextOrder(Input):
    ids: list[str] = Field(max_length=100)


class EmbeddingRequest(Input):
    embedding_profile_id: str | None = Field(default=None, min_length=1, max_length=100)


class DocumentTitleInput(Input):
    title: str = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def meaningful(self):
        if not self.title.strip():
            raise ValueError("Document title cannot be blank.")
        return self


class RevisionInput(EmbeddingRequest):
    document_id: str | None = Field(default=None, min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    filename: str | None = Field(default=None, min_length=1, max_length=500)
    version_number: int | None = Field(default=None, gt=0)
    content: str = Field(min_length=1, max_length=100_000)
    title: str = Field(min_length=1, max_length=200)


class DocumentInput(EmbeddingRequest):
    filename: str | None = Field(default=None, min_length=1, max_length=500)
    document_id: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    title: str = Field(default="", max_length=200)
    source: str = Field(default="", max_length=500)
    version: str = Field(min_length=1, max_length=80)
    content: str = Field(min_length=1, max_length=100_000, validation_alias=AliasChoices("content", "body"))
    metadata: dict[str, str] = Field(default_factory=dict)
    previous_version_id: str | None = Field(default=None, min_length=1, max_length=100)
    sequence_number: int | None = Field(default=None, gt=0)
    published_on: date | None = None
    effective_on: date | None = None

    @field_validator("version", mode="before")
    @classmethod
    def numeric_version(cls, value):
        return str(value) if type(value) is int else value

    @model_validator(mode="after")
    def document_fields(self):
        self.source = self.source or self.filename or f"{self.document_id}-v{self.version}.txt"
        self.filename = self.filename or self.source.replace("\\", "/").rsplit("/", 1)[-1]
        self.title = self.title or self.filename
        return self


class ExperimentDocument(Input):
    document_id: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    version: str = Field(default="1", min_length=1, max_length=80)
    filename: str = Field(default="", max_length=500)
    title: str = Field(default="", max_length=200)
    source: str = Field(default="", max_length=500)
    content: str = Field(default="", max_length=100_000)
    metadata: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True


class QuestionInput(Input):
    answer_prompt_id: str | None = Field(default=None, min_length=1, max_length=100)
    documents: list[ExperimentDocument] | None = Field(default=None, max_length=100)
    output_mode: Literal["custom", "facts_and_answer"] = "custom"
    answer_json: dict | None = None
    system_prompt: str | None = Field(default=None, max_length=20_000)
    llm_context: str | None = Field(default=None, max_length=200_000)
    context_chunk_ids: list[str] | None = Field(default=None, max_length=2000)
    context_override: str | None = Field(default=None, max_length=100_000)
    context_ids: list[str] | None = Field(default=None, max_length=100)
    question: str = Field(min_length=1, max_length=2000)
    mode: Literal["experiment"] = "experiment"
    top_k: int = Field(default=4, ge=1, le=20)
    context_mode: Literal["retrieval", "explicit"] = "retrieval"
    version_instructions: bool = False
    prompt_instructions: str = Field(default="", max_length=10_000)
    version_ids: list[str] | None = Field(default=None, max_length=100)
    temperature: float = Field(default=0, ge=0, le=2)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    max_tokens: int = Field(default=400, ge=16, le=8192)
    llm_profile_id: str | None = Field(default=None, min_length=1, max_length=100)
    embedding_profile_id: str | None = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode="after")
    def meaningful_question(self):
        if not self.question.strip():
            raise ValueError("A question cannot contain only whitespace.")
        if self.documents is None and self.version_ids is not None and not self.version_ids:
            raise ValueError("Select at least one document version.")
        if self.version_ids and len(set(self.version_ids)) != len(self.version_ids):
            raise ValueError("Select each document version only once.")
        if self.context_mode == "explicit" and not self.version_ids and self.documents is None:
            raise ValueError("Explicit context requires an ordered selection of document versions.")
        return self


class ExperimentInput(Input):
    answer_prompt_id: str | None = Field(default=None, min_length=1, max_length=100)
    review_facts_prompt_id: str | None = Field(default=None, min_length=1, max_length=100)
    review_answer_prompt_id: str | None = Field(default=None, min_length=1, max_length=100)
    system_prompt: str | None = Field(default=None, max_length=20_000)
    llm_context: str | None = Field(default=None, max_length=200_000)
    context_chunk_ids: list[str] | None = Field(default=None, max_length=2000)
    context_override: str | None = Field(default=None, max_length=100_000)
    question: str | None = Field(default=None, min_length=1, max_length=2000)
    context_mode: Literal["retrieval", "explicit"] | None = None
    version_ids: list[str] | None = Field(default=None, min_length=1, max_length=100)
    top_k: int | None = Field(default=None, ge=1, le=20)
    version_instructions: bool | None = None
    prompt_instructions: str | None = Field(default=None, max_length=10_000)
    context_ids: list[str] | None = Field(default=None, max_length=100)
    mode: Literal["experiment"] = "experiment"
    repetitions: int = Field(default=3, ge=1, le=100)
    llm_profile_id: str | None = Field(default=None, min_length=1, max_length=100)
    embedding_profile_id: str | None = Field(default=None, min_length=1, max_length=100)
    temperature: float = Field(default=0, ge=0, le=2)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    max_tokens: int = Field(default=1024, ge=16, le=8192)


class FactSource(Input):
    quote: str | None = Field(default=None, min_length=1, max_length=100_000)

    @model_serializer(mode="wrap")
    def serialize_source(self, handler):
        value = handler(self)
        if value["quote"] is None:
            del value["quote"]
        return value

    document_id: str = Field(min_length=1, max_length=100)
    version: str = Field(min_length=1, max_length=80)


class ExpectedFact(Input):
    optional: bool = False
    fact: str = Field(min_length=1, max_length=5000)
    sources: list[FactSource] = Field(min_length=1, max_length=100)


class SemanticReference(Input):
    facts: list[str | ExpectedFact] = Field(default_factory=list, max_length=100)
    answer: str | None = Field(
        default=None, max_length=10_000, validation_alias=AliasChoices("answer", "conclusion")
    )

    @field_validator("facts")
    @classmethod
    def meaningful_facts(cls, values):
        if any(isinstance(value, str) and (not value.strip() or len(value) > 5000) for value in values):
            raise ValueError("Expected facts must be nonempty and at most 5,000 characters.")
        return values


class SemanticEvaluatorInput(Input):
    confidence_threshold: float | None = Field(default=90, ge=0, le=100, allow_inf_nan=False)
    llm_profile_id: str = Field(min_length=1, max_length=100)
    temperature: float = Field(default=0, ge=0, le=2)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    max_tokens: int = Field(default=4096, ge=128, le=8192)


class ExperimentDefinition(Input):
    """Bewerkbare proef; validatiegegevens zijn nooit modelinvoer."""

    output_mode: Literal["custom", "facts_and_answer"] = "custom"
    semantic_reference: SemanticReference | None = None
    semantic_evaluator: SemanticEvaluatorInput | None = None
    title: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=2000)
    version_ids: list[str] = Field(default_factory=list, max_length=100)
    documents: list[ExperimentDocument] | None = Field(default=None, max_length=100)
    system_prompt: str = Field(default="", max_length=20_000)
    context: str = Field(default="", max_length=10_000)
    answer_json: dict = Field(default_factory=dict)
    validation_json: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_definition(self):
        from traceability_lab.defined_experiments import validate_contract

        if not self.title.strip() or not self.question.strip():
            raise ValueError("Title and question cannot be blank.")
        if self.documents is None and not self.version_ids:
            raise ValueError("An experiment needs documents.")
        if self.documents is not None:
            if self.version_ids:
                raise ValueError("Use owned documents or legacy version_ids, not both.")
            keys = [(doc.document_id, doc.version) for doc in self.documents]
            if len(keys) != len(set(keys)):
                raise ValueError("Document ID and version must be unique within the experiment.")
        if len(set(self.version_ids)) != len(self.version_ids):
            raise ValueError("Select each document version only once.")
        if self.output_mode == "facts_and_answer":
            from traceability_lab.fact_answers import FACT_ANSWER_JSON

            self.answer_json = FACT_ANSWER_JSON.copy()
            if self.validation_json:
                raise ValueError(
                    "Use semantic_reference for expected facts and answer, not exact Validation JSON."
                )
        else:
            if self.semantic_evaluator is not None or self.semantic_reference is not None:
                raise ValueError("Semantic evaluation is available only for facts_and_answer.")
            validate_contract(self.answer_json, self.validation_json)
        return self


class DefinedExperimentRun(Input):
    definition: ExperimentDefinition
    execution: ExperimentInput = Field(default_factory=ExperimentInput)
    saved_id: str | None = None


class ArtifactVerificationInput(Input):
    path: str = Field(min_length=1, max_length=4096)
    pid: int | None = Field(default=None, ge=1)


class DeploymentManifest(Input):
    size_bytes: int | None = Field(default=None, ge=4)
    verification: dict | None = None

    @model_validator(mode="before")
    @classmethod
    def portable_manifest(cls, value):
        if not isinstance(value, dict) or not ("artifact" in value or "runtime" in value):
            return value
        data = dict(value)
        artifact, runtime = data.pop("artifact", {}), data.pop("runtime", {})
        if not isinstance(artifact, dict) or not isinstance(runtime, dict):
            raise ValueError("Artifact and runtime must be JSON objects.")
        if set(artifact) - {"filename", "sha256", "size_bytes", "architecture", "quantization"} or set(
            runtime
        ) - {"engine", "revision", "configuration"}:
            raise ValueError("Unknown artifact or runtime manifest field.")
        fields = dict(artifact)
        for key, target in (
            ("engine", "engine"),
            ("revision", "engine_revision"),
            ("configuration", "runtime_configuration"),
        ):
            if key in runtime:
                fields[target] = runtime[key]
        for key, item in fields.items():
            if key in data and data[key] != item:
                raise ValueError(f"Conflicting manifest field: {key}.")
            data[key] = item
        data.setdefault("provenance", "imported deployment manifest")
        return data

    filename: str | None = Field(default=None, max_length=500)
    sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    architecture: str | None = Field(default=None, max_length=200)
    quantization: str | None = Field(default=None, max_length=100)
    engine: str | None = Field(default=None, max_length=100)
    engine_revision: str | None = Field(default=None, max_length=200)
    runtime_configuration: dict = Field(default_factory=dict)
    provenance: str = Field(min_length=1, max_length=500)

    @field_validator("runtime_configuration")
    @classmethod
    def serializable_runtime(cls, value):
        json.dumps(value, allow_nan=False)
        return value


class ProfileInput(Input):
    """Een menselijk label plus de daadwerkelijke API-identiteit en onderzoeksmetadata."""

    deployment_manifest: DeploymentManifest | None = None
    capture_server_metadata: bool | None = None
    trust_server_identity: bool = False
    name: str = Field(min_length=1, max_length=100)
    kind: Literal["llm", "embedding"]
    url: str = Field(min_length=1, max_length=500)
    model: str = Field(min_length=1, max_length=200)
    max_tokens: int | None = Field(default=None, ge=128, le=8192)
    dimensions: int | None = Field(default=None, ge=1, le=16384)
    revision: str | None = Field(default=None, max_length=200)
    server_revision: str | None = Field(default=None, max_length=200)
    artifact_sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    precision: str | None = Field(default=None, max_length=100)
    hardware: str | None = Field(default=None, max_length=200)
    timeout_seconds: float = Field(default=120, gt=0, le=600)
    enable_thinking: bool | None = None
    query_prefix: str = Field(default="", max_length=1000)
    document_prefix: str = Field(default="", max_length=1000)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value):
        value = service_url(value)
        if not value.endswith("/v1"):
            raise ValueError("Use the API base URL including /v1.")
        return value

    @model_validator(mode="after")
    def validate_profile(self):
        if self.trust_server_identity and self.capture_server_metadata is not True:
            raise ValueError("Trusted server identity requires live server metadata capture.")
        if not self.name.strip() or not self.model.strip():
            raise ValueError("Name and model ID cannot contain only whitespace.")
        if self.kind == "embedding" and self.dimensions is None:
            raise ValueError("Specify the expected embedding dimensions.")
        if self.kind == "llm" and (self.dimensions is not None or self.query_prefix or self.document_prefix):
            raise ValueError("Dimensions and embedding prefixes apply only to embedding profiles.")
        return self


class ClaimInput(Input):
    # Halfopen Python-tekenoffsets in het antwoord: answer[start:end].
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    annotator: str = Field(min_length=1, max_length=100)


class AssessmentInput(Input):
    classification: Literal["supported", "contradicted", "insufficient_evidence"]
    scope: Literal["retrieved", "corpus"] = "retrieved"
    chunk_id: str | None = None
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, gt=0)
    annotator: str = Field(min_length=1, max_length=100)
    note: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def evidence_required(self):
        if self.chunk_id is None:
            if (
                self.classification != "insufficient_evidence"
                or self.start is not None
                or self.end is not None
            ):
                raise ValueError("An exact source span is required for support or contradiction.")
        elif self.start is None or self.end is None or self.end <= self.start:
            raise ValueError("Specify the start and end of the evidence span.")
        return self


class ReviewInput(Input):
    """Beoordeling van observeerbaar gedrag; onbekend blijft null, niet false."""

    answer_correct: bool | None = None
    abstained: bool | None = None
    conflict_detected: bool | None = None
    claims_complete: bool = False
    annotator: str = Field(min_length=1, max_length=100)
    note: str = Field(min_length=1, max_length=2000)
