"""Explicit test fixtures, never application defaults."""

from traceability_lab.db import Database, json_text, now
from traceability_lab.models import ProfileInput


def seed_test_profiles(db: Database):
    """Alleen bij eerste installatie invullen; handmatige aanpassingen behouden."""
    presets = {
        "test-reviewer": ProfileInput(
            name="Test reviewer",
            kind="llm",
            url="http://127.0.0.1:8002/v1",
            model="test-reviewer-model",
            precision="Q5_K_M (server metadata)",
            timeout_seconds=180,
        ),
        "test-generator": ProfileInput(
            name="Test generator",
            kind="llm",
            url="http://127.0.0.1:8000/v1",
            model="test-generator-model",
            hardware="GPU-server",
            precision="Q4_K_M (servermelding)",
            enable_thinking=False,
            timeout_seconds=180,
        ),
        "test-embedding": ProfileInput(
            name="Test embeddings",
            kind="embedding",
            url="http://127.0.0.1:8001/v1",
            model="bge-m3",
            dimensions=1024,
            hardware="GPU-server",
            precision="FP16",
            timeout_seconds=60,
        ),
    }
    with db.connect() as connection:
        for identity, data in presets.items():
            timestamp = now()
            connection.execute(
                "INSERT OR IGNORE INTO model_profiles VALUES (?,?,?,?)",
                (identity, json_text(data.model_dump()), timestamp, timestamp),
            )
