"""Elke test krijgt een eigen database; geen modeldownload of live server nodig."""

import pytest
from fastapi.testclient import TestClient

from traceability_lab.config import Settings
from traceability_lab.db import Database
from traceability_lab.main import create_app


@pytest.fixture
def settings(tmp_path):
    return Settings(
        backend="demo",
        database=tmp_path / "lab.sqlite3",
        experiments_directory=tmp_path / "no-examples",
        otlp_endpoint=None,
        runtime_metadata_default=False,
    )


@pytest.fixture
def db(settings):
    return Database(settings.database)


@pytest.fixture
def client(settings):
    from scenario_fixtures import bootstrap, install_scenarios

    app = create_app(settings)
    install_scenarios(app, settings)
    from legacy_api import install_legacy_api

    install_legacy_api(app, settings)
    with TestClient(app) as client:
        bootstrap(Database(settings.database))
        from profile_presets import seed_test_profiles

        seed_test_profiles(Database(settings.database))
        yield client


@pytest.fixture
def completed(client):
    result = client.post("/test-scenarios/complete/run", json={"repetitions": 1})
    assert result.status_code == 201, result.text
    identity = result.json()["run_ids"][0]
    return client.get(f"/api/runs/{identity}").json()


def pytest_addoption(parser):
    parser.addoption("--browser", action="store_true", help="Run isolated Playwright browser regressions.")
