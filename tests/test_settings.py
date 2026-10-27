"""Local environment files do not change the fixed application defaults."""

from traceability_lab.config import PROJECT_ROOT, Settings


def test_environment_cannot_redirect_app_data(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("LAB_DATABASE=unexpected.sqlite3\nLAB_BACKEND=local\n")
    monkeypatch.setenv("LAB_DATABASE", "another.sqlite3")
    monkeypatch.setenv("LAB_BACKEND", "local")
    settings = Settings()
    assert settings.database == PROJECT_ROOT / "data/lab.sqlite3"
    assert settings.experiments_directory == PROJECT_ROOT / "experiments"
    assert settings.backend == "demo"
    assert Settings(database=tmp_path / "explicit.sqlite3").database == tmp_path / "explicit.sqlite3"


def test_runner_defaults_match_the_ui():
    from traceability_lab.models import ExperimentInput

    assert ExperimentInput().repetitions == 3
