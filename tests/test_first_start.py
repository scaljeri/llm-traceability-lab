"""Examples are imported once, when the database file does not exist."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from traceability_lab.main import create_app

EXAMPLES = Path(__file__).resolve().parents[1] / "experiments"


def test_first_start_imports_all_examples_and_restart_preserves_edits(settings):
    settings = settings.model_copy(update={"experiments_directory": EXAMPLES})
    with TestClient(create_app(settings)) as client:
        saved = client.get("/api/experiment-definitions").json()
        assert len(saved) == len(list(EXAMPLES.glob("*.json"))) == 7
        assert [item["definition"]["title"][:2] for item in saved] == [f"{i:02}" for i in range(1, 8)]
        assert all(item["definition"]["documents"] for item in saved)
        assert all(item["definition"]["semantic_evaluator"] is None for item in saved)
        definition = saved[0]["definition"]
        definition["title"] = "My edited example"
        assert client.put(f"/api/experiment-definitions/{saved[0]['id']}", json=definition).status_code == 200
        assert client.delete(f"/api/experiment-definitions/{saved[1]['id']}").status_code == 200
    with TestClient(create_app(settings)) as client:
        restarted = client.get("/api/experiment-definitions").json()
        assert len(restarted) == 6
        assert any(item["definition"]["title"] == "My edited example" for item in restarted)
        assert saved[1]["id"] not in {item["id"] for item in restarted}
        assert client.get("/api/runs").json() == []


def test_existing_empty_database_is_not_seeded(settings):
    from traceability_lab.db import Database

    Database(settings.database).initialize()
    settings = settings.model_copy(update={"experiments_directory": EXAMPLES})
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/experiment-definitions").json() == []


def test_invalid_example_does_not_leave_a_partly_initialized_database(settings, tmp_path):
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "01-valid.json").write_bytes(next(EXAMPLES.glob("01-*.json")).read_bytes())
    (examples / "02-invalid.json").write_text('{"title": "Invalid"}')
    settings = settings.model_copy(update={"experiments_directory": examples})
    with pytest.raises(ValueError):
        with TestClient(create_app(settings)):
            pass
    assert not settings.database.exists()
