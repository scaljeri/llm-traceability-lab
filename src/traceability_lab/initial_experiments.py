"""Load owned-document examples only when creating a database for the first time."""

import json
from pathlib import Path
from uuid import uuid4

from traceability_lab.db import Database, json_text, now
from traceability_lab.models import ExperimentDefinition
from traceability_lab.transfers import PortableDocument, PortableExperiment


def read_initial_experiments(directory: Path) -> list[ExperimentDefinition]:
    definitions = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "format" in payload:
                portable = PortableExperiment.model_validate(payload)
                if portable.format != "traceability-experiment-v2":
                    raise ValueError("Initial experiments require self-contained v2 documents.")
                payload = dict(portable.experiment)
                documents = []
                for doc in payload.get("documents", []):
                    document = PortableDocument.model_validate(
                        {key: value for key, value in doc.items() if key != "enabled"}
                    )
                    documents.append(
                        {**document.model_dump(exclude={"id", "sha256"}), "enabled": doc.get("enabled", True)}
                    )
                payload["documents"] = documents
            definition = ExperimentDefinition.model_validate(payload)
            if definition.documents is None:
                raise ValueError("Initial experiments must contain their own documents.")
            definitions.append(definition)
        except (ValueError, TypeError, AttributeError) as error:
            raise ValueError(f"Cannot load initial experiment {path.name}: {error}") from error
    return definitions


def store_initial_experiments(db: Database, definitions: list[ExperimentDefinition]) -> None:
    # One transaction prevents a partial set of saved examples.
    with db.connect() as connection:
        for definition in definitions:
            timestamp = now()
            connection.execute(
                "INSERT INTO experiment_definitions VALUES (?,?,?,?,?)",
                ("saved-" + uuid4().hex, 1, json_text(definition.model_dump()), timestamp, timestamp),
            )
