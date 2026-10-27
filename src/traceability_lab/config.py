"""Vaste appstandaarden; expliciete overrides voor tests en serverprofielen."""

from ipaddress import ip_address, ip_network
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

PACKAGE = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE.parents[1]


def service_url(value: str, *, allow_lan: bool = True) -> str:
    """Modelservers zijn expliciete loopback- of RFC1918-adressen, geen willekeurige URLs."""
    url = urlparse(value)
    allowed = url.hostname in {"127.0.0.1", "localhost", "::1"}
    if allow_lan and url.hostname:
        try:
            address = ip_address(url.hostname)
            allowed = allowed or any(
                address in ip_network(network)
                for network in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
            )
        except ValueError:
            pass
    if url.scheme not in {"http", "https"} or not allowed:
        raise ValueError("Use an http(s) URL on loopback or a private IP address.")
    if url.username or url.password or url.query or url.fragment:
        raise ValueError("A service URL cannot contain credentials, a query or a fragment.")
    # Ook een ongeldige poort vroeg afwijzen.
    _ = url.port
    return value.rstrip("/")


class Settings(BaseModel):
    prompts_directory: Path = PROJECT_ROOT / "prompts"
    model_config = ConfigDict(extra="forbid")

    backend: Literal["demo", "local"] = "demo"
    database: Path = PROJECT_ROOT / "data/lab.sqlite3"
    experiments_directory: Path = PROJECT_ROOT / "experiments"
    embedding_path: Path | None = None
    embedding_url: str | None = None
    embedding_model: str = "bge-m3"
    embedding_dimensions: int = Field(default=1024, ge=1, le=16384)
    embedding_revision: str | None = None
    embedding_server_revision: str | None = None
    embedding_artifact_sha256: str | None = None
    embedding_precision: str | None = None
    embedding_hardware: str | None = None
    embedding_timeout_seconds: float = Field(default=60, gt=0, le=600)
    query_prefix: str = ""
    document_prefix: str = ""
    llm_url: str = "http://127.0.0.1:8080/v1"
    llm_model: str = "local-model"
    runtime_metadata_default: bool = True
    llm_artifact_sha256: str | None = None
    llm_server_revision: str | None = None
    llm_enable_thinking: bool | None = None
    code_revision: str | None = None
    timeout_seconds: float = Field(default=120, gt=0, le=600)
    otlp_endpoint: str | None = None
    jaeger_url: str | None = None

    @field_validator("llm_url", "embedding_url", "otlp_endpoint", "jaeger_url")
    @classmethod
    def require_local_service(cls, value: str | None, info: ValidationInfo) -> str | None:
        """De LLM mag op het privénetwerk draaien; trace-export blijft op loopback."""
        if value is None:
            return value
        return service_url(value, allow_lan=info.field_name in {"llm_url", "embedding_url"})
