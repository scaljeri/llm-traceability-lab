"""Twee expliciete embeddingpaden; nooit stil terugvallen op de oefenstand."""

import hashlib
import math
import re
import threading
from importlib.metadata import version
from uuid import uuid4

import httpx

from traceability_lab.config import Settings
from traceability_lab.db import json_text
from traceability_lab.documents import digest


def artifact_hash(directory) -> str:
    """Fingerprint van bestandsnamen en inhoud van de lokale modelsnapshot."""
    manifest = []
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(directory).parts
    )
    if not files:
        raise ValueError("The embedding directory contains no model files.")
    for path in files:
        result = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                result.update(block)
        manifest.append((path.relative_to(directory).as_posix(), result.hexdigest()))
    return digest(json_text(manifest))


class Embedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._configuration = None
        self._lock = threading.Lock()
        self.observations = []
        self._cache_scope = uuid4().hex

    def configuration(self) -> dict:
        """Het model wordt pas geladen als iemand daadwerkelijk een run uitvoert."""
        with self._lock:
            if self._configuration is None:
                if self.settings.backend == "demo":
                    self._configuration = {
                        "id": "demo-token-hash-v1",
                        "dimensions": 256,
                        "normalized": True,
                        "kind": "deterministic_test_double",
                        "revision": "sha256-tokens-v1",
                    }
                elif self.settings.embedding_url:
                    pinned = bool(self.settings.embedding_revision or self.settings.embedding_artifact_sha256)
                    self._configuration = {
                        "id": self.settings.embedding_model,
                        "kind": "openai-compatible-embeddings",
                        "url": self.settings.embedding_url,
                        "dimensions": self.settings.embedding_dimensions,
                        "revision": self.settings.embedding_revision,
                        "server_revision": self.settings.embedding_server_revision,
                        "artifact_sha256": self.settings.embedding_artifact_sha256,
                        "artifact_identity_status": "operator_declared" if pinned else "unknown",
                        "precision": self.settings.embedding_precision,
                        "hardware": self.settings.embedding_hardware,
                        "normalized": True,
                        "client_normalization": "l2-v1",
                        "query_prefix": self.settings.query_prefix,
                        "document_prefix": self.settings.document_prefix,
                        # Zonder versie-identiteit kan een alias op de server veranderen.
                        # Daarom dan uitsluitend binnen deze run vectoren hergebruiken.
                        "cache_scope": "pinned-configuration" if pinned else self._cache_scope,
                    }
                else:
                    path = self.settings.embedding_path
                    if path is None or not path.is_dir():
                        raise ValueError(
                            "Configure a local sentence-transformers model directory through Settings.embedding_path."
                        )
                    try:
                        from sentence_transformers import SentenceTransformer
                    except ImportError as error:
                        raise ValueError("Install local model support with: uv sync --extra local") from error
                    fingerprint = artifact_hash(path)
                    self._model = SentenceTransformer(
                        str(path), device="cpu", local_files_only=True, trust_remote_code=False
                    )
                    self._configuration = {
                        "id": str(path.resolve()),
                        "artifact_sha256": fingerprint,
                        "dimensions": self._model.get_sentence_embedding_dimension(),
                        "max_seq_length": self._model.max_seq_length,
                        "device": "cpu",
                        "normalized": True,
                        "sentence_transformers": version("sentence-transformers"),
                        "torch": version("torch"),
                        "kind": "sentence-transformers",
                    }
            return {**self._configuration, "document_input_format": "document-id-and-chunk-v1"}

    def key(self) -> str:
        return digest(json_text(self.configuration()))

    def encode(self, text: str, *, role: str = "document") -> list[float]:
        configuration = self.configuration()
        if configuration["kind"] == "openai-compatible-embeddings":
            prefix = self.settings.query_prefix if role == "query" else self.settings.document_prefix
            body = {
                "model": self.settings.embedding_model,
                "input": prefix + text,
                "encoding_format": "float",
            }
            with httpx.Client(
                timeout=self.settings.embedding_timeout_seconds, trust_env=False, follow_redirects=False
            ) as client:
                response = client.post(self.settings.embedding_url + "/embeddings", json=body)
                response.raise_for_status()
                payload = response.json()
            try:
                data = payload["data"]
                if len(data) != 1 or data[0].get("index", 0) != 0:
                    raise ValueError("Expected exactly one embedding for one text.")
                values = data[0]["embedding"]
            except (KeyError, IndexError, TypeError, AttributeError) as error:
                raise ValueError("The embedding server returned an invalid response.") from error
            if (
                not isinstance(values, list)
                or len(values) != configuration["dimensions"]
                or not all(type(value) in (int, float) and math.isfinite(value) for value in values)
            ):
                raise ValueError(
                    f"Expected {configuration['dimensions']} finite numbers from the embedding server."
                )
            length = math.hypot(*values)
            if not length or not math.isfinite(length):
                raise ValueError("The embedding server returned an empty or invalid vector.")
            self.observations.append(
                {
                    "role": role,
                    "input_sha256": digest(prefix + text),
                    "requested_model": self.settings.embedding_model,
                    "reported_model": payload.get("model"),
                    "usage": payload.get("usage"),
                    "dimensions": len(values),
                }
            )
            return [value / length for value in values]
        if self.settings.backend == "demo":
            # Lexicale hashing is alleen een stabiele testdouble, geen semantisch model.
            vector = [0.0] * configuration["dimensions"]
            for token in re.findall(r"\w+", text.casefold()):
                index = int(digest(token)[:8], 16) % len(vector)
                vector[index] += 1
            if not any(vector):
                vector[0] = 1
            length = math.sqrt(sum(value * value for value in vector))
            return [value / length for value in vector]
        with self._lock:
            values = self._model.encode(text, normalize_embeddings=True, show_progress_bar=False).tolist()
        if len(values) != configuration["dimensions"] or not all(math.isfinite(v) for v in values):
            raise ValueError("The embedding model returned an invalid vector.")
        return values
