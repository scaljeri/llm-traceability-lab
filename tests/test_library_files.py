"""Explicit file sync uses temporary folders and never alters bundled examples."""

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from traceability_lab.config import PROJECT_ROOT, Settings
from traceability_lab.db import Database
from traceability_lab.library_files import replace_json_files
from traceability_lab.main import create_app


@pytest.fixture
def library(tmp_path):
    for folder in ("experiments", "prompts"):
        shutil.copytree(PROJECT_ROOT / folder, tmp_path / folder)
    settings = Settings(
        database=tmp_path / "data/lab.sqlite3",
        experiments_directory=tmp_path / "experiments",
        prompts_directory=tmp_path / "prompts",
        runtime_metadata_default=False,
    )
    with TestClient(create_app(settings)) as client:
        yield client, settings, Database(settings.database)


def files(directory):
    return {path.name: path.read_bytes() for path in directory.glob("*.json")}


def test_export_names_invalid_saved_prompt_without_changing_files(library):
    client, settings, db = library
    before = [files(path) for path in (settings.experiments_directory, settings.prompts_directory)]
    value = next(item for item in client.get("/api/prompts").json() if item["type"] == "answer")
    value["user_prompt"] = "{{user_question}}"
    with db.connect() as connection:
        connection.execute("UPDATE prompts SET payload=? WHERE id=?", (json.dumps(value), value["id"]))
    response = client.post("/api/library-files/export")
    assert response.status_code == 422
    message = response.json()["detail"]
    assert value["name"] in message and value["id"] in message
    assert "{{documents}}" in message
    assert "input_value" not in message and "pydantic.dev" not in message
    assert before == [files(path) for path in (settings.experiments_directory, settings.prompts_directory)]


def change_json(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def test_import_replaces_both_libraries_and_preserves_history_and_servers(library):
    client, settings, db = library
    profile = client.post(
        "/api/profiles",
        json={"name": "My server", "kind": "llm", "url": "http://localhost:8080/v1", "model": "test"},
    ).json()
    with db.connect() as connection:
        connection.execute(
            "INSERT INTO runs(id,created_at,status,trace_id,question,mode,backend,snapshot,messages,answer) "
            "VALUES ('history','today','completed','trace','Question','experiment','demo','{}','[]','Original answer')"
        )
    original_run = db.one("SELECT * FROM runs WHERE id='history'")
    (settings.experiments_directory / "01-complete-experiment.json").unlink()
    change_json(
        settings.experiments_directory / "02-missing-information.json",
        lambda value: value["experiment"].update(title="Changed experiment", documents=[]),
    )
    change_json(
        settings.prompts_directory / "generation_prompt.json",
        lambda value: value.update(name="Changed prompt"),
    )
    new = {
        "title": "New file",
        "question": "New question?",
        "documents": [],
        "output_mode": "facts_and_answer",
    }
    (settings.experiments_directory / "new.json").write_text(json.dumps(new))
    assert (
        client.post(
            "/api/prompts",
            json={
                "id": "app-only",
                "type": "answer",
                "name": "App only",
                "user_prompt": "{{user_question}}\n{{documents}}",
            },
        ).status_code
        == 201
    )
    response = client.post("/api/library-files/import")
    assert response.status_code == 200, response.text
    assert response.json() == {"experiments": 7, "prompts": 2}
    saved = [item["definition"] for item in client.get("/api/experiment-definitions").json()]
    assert any(item["title"] == "New file" for item in saved)
    assert next(item for item in saved if item["title"] == "Changed experiment")["documents"] == []
    assert not any(item["title"].startswith("01 -") for item in saved)
    assert {item["name"] for item in client.get("/api/prompts").json()} == {
        "Changed prompt",
        "review_prompt",
    }
    assert db.one("SELECT * FROM runs WHERE id='history'") == original_run
    assert client.get("/api/profiles").json() == [profile]
    assert client.post("/api/library-files/import").json() == response.json()


@pytest.mark.parametrize("invalid", ["experiment", "prompt", "prompt-shape", "review-stage"])
def test_import_validates_every_file_before_changing_database(library, invalid):
    client, settings, _ = library
    before = [client.get(path).json() for path in ("/api/experiment-definitions", "/api/prompts")]
    if invalid == "experiment":
        (settings.experiments_directory / "broken.json").write_text('{"question": 17}')
    elif invalid == "prompt":
        (settings.prompts_directory / "broken.json").write_text("{")
    elif invalid == "prompt-shape":
        (settings.prompts_directory / "broken.json").write_text("[]")
    else:
        (settings.prompts_directory / "fact_review_prompt.json").unlink()
    response = client.post("/api/library-files/import")
    assert response.status_code == 422, response.text
    assert [client.get(path).json() for path in ("/api/experiment-definitions", "/api/prompts")] == before


def test_export_round_trip_all_documents_and_prompt_stages(library):
    client, settings, _ = library
    saved = client.get("/api/experiment-definitions").json()[0]
    definition = saved["definition"]
    definition["title"] = "../../An edited experiment"
    definition["documents"][0]["enabled"] = False
    assert client.put(f"/api/experiment-definitions/{saved['id']}", json=definition).status_code == 200
    note = settings.experiments_directory / "keep.txt"
    note.write_text("Keep this file")
    stale = settings.prompts_directory / "stale.json"
    stale.write_text("Invalid old JSON can be overwritten")
    response = client.post("/api/library-files/export")
    assert response.status_code == 200, response.text
    assert response.json() == {"experiments": 7, "prompts": 2}
    assert note.read_text() == "Keep this file"
    assert not stale.exists()
    assert set(files(settings.prompts_directory)) == {
        "generation_prompt.json",
        "fact_review_prompt.json",
        "answer_review_prompt.json",
    }
    experiment = json.loads((settings.experiments_directory / "an-edited-experiment.json").read_text())[
        "experiment"
    ]
    assert experiment["documents"][0]["enabled"] is False
    assert experiment["documents"][0]["content"] == definition["documents"][0]["content"]
    assert "system_prompt" not in experiment
    expected_prompts = client.get("/api/prompts").json()
    assert client.post("/api/library-files/import").status_code == 200
    assert client.get("/api/prompts").json() == expected_prompts
    again = next(
        item["definition"]
        for item in client.get("/api/experiment-definitions").json()
        if item["definition"]["title"] == definition["title"]
    )
    assert again["documents"] == definition["documents"]
    assert again["semantic_reference"] == definition["semantic_reference"]


@pytest.mark.parametrize("direction", ["import", "export"])
def test_sync_is_blocked_during_an_active_run(library, direction, monkeypatch):
    client, _, _ = library
    with monkeypatch.context() as patch:
        patch.setattr(client.app.state.executions, "jobs", {"busy": {"state": {"status": "running"}}})
        assert client.post(f"/api/library-files/{direction}").status_code == 409


def test_failed_export_restores_original_json_in_both_folders(tmp_path, monkeypatch):
    first, second = tmp_path / "experiments", tmp_path / "prompts"
    for directory in (first, second):
        directory.mkdir()
        (directory / "original.json").write_text('{"original": true}')
    before = [files(first), files(second)]
    replace = Path.replace
    failed = False

    def fail_once(source, target):
        nonlocal failed
        if source.parent.name == "incoming" and target.parent == second and not failed:
            failed = True
            raise OSError("Simulated failure while installing second library")
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_once)
    with pytest.raises(OSError, match="Simulated"):
        replace_json_files({first: {"new.json": {}}, second: {"new.json": {}}})
    assert [files(first), files(second)] == before
    assert not list(tmp_path.glob("*/.sync-*"))


def test_database_error_rolls_back_both_imports(library):
    client, _, db = library
    before = [client.get(path).json() for path in ("/api/experiment-definitions", "/api/prompts")]
    with db.connect() as connection:
        connection.execute(
            "CREATE TRIGGER reject_prompt BEFORE INSERT ON prompts BEGIN SELECT RAISE(ABORT,'test'); END"
        )
    assert client.post("/api/library-files/import").status_code == 409
    assert [client.get(path).json() for path in ("/api/experiment-definitions", "/api/prompts")] == before


def test_multiple_prompt_sets_and_duplicate_titles_round_trip(library):
    client, settings, _ = library
    for prompt in client.get("/api/prompts").json():
        assert client.post("/api/prompts", json={**prompt, "id": prompt["id"] + "-copy"}).status_code == 201
    definition = client.get("/api/experiment-definitions").json()[0]["definition"]
    assert client.post("/api/experiment-definitions", json=definition).status_code == 201
    expected = client.get("/api/prompts").json()
    assert client.post("/api/library-files/export").json() == {"experiments": 8, "prompts": 4}
    assert len(files(settings.experiments_directory)) == 8
    assert len(files(settings.prompts_directory)) == 6
    assert client.post("/api/library-files/import").json() == {"experiments": 8, "prompts": 4}
    assert client.get("/api/prompts").json() == expected


def test_export_staging_failure_leaves_both_folders_unchanged(library, monkeypatch):
    client, settings, _ = library
    directories = (settings.experiments_directory, settings.prompts_directory)
    before = [files(directory) for directory in directories]
    write = Path.write_text

    def fail_prompt_write(path, *args, **kwargs):
        if path.parent.name == "incoming" and path.parents[2] == settings.prompts_directory:
            raise PermissionError("Read-only prompt mount")
        return write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_prompt_write)
    response = client.post("/api/library-files/export")
    assert response.status_code == 400, response.text
    assert "Read-only" in response.json()["detail"]
    assert [files(directory) for directory in directories] == before


def test_export_rejects_filenames_that_collide_on_case_insensitive_disks(tmp_path):
    with pytest.raises(ValueError, match="collide"):
        replace_json_files({tmp_path: {"answer-A.json": {}, "answer-a.json": {}}})
    assert not files(tmp_path)


@pytest.mark.parametrize("direction", ["import", "export"])
@pytest.mark.parametrize("alias", [False, True])
def test_sync_refuses_identical_directories_without_data_loss(library, direction, alias, tmp_path):
    client, settings, _ = library
    before_files = files(settings.experiments_directory)
    before = [client.get(path).json() for path in ("/api/experiment-definitions", "/api/prompts")]
    path = settings.experiments_directory
    if alias:
        path = tmp_path / "alias"
        path.symlink_to(settings.experiments_directory, target_is_directory=True)
    settings.prompts_directory = path
    response = client.post(f"/api/library-files/{direction}")
    assert response.status_code == 422, response.text
    assert "different directories" in response.json()["detail"]
    assert files(settings.experiments_directory) == before_files
    assert [client.get(path).json() for path in ("/api/experiment-definitions", "/api/prompts")] == before
