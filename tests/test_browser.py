"""Opt-in browser suite with isolated databases; never sends requests to a real LLM."""

import importlib
import socket
import threading
import time

import pytest
import uvicorn

from traceability_lab.config import PROJECT_ROOT, Settings
from traceability_lab.db import Database
from traceability_lab.initial_experiments import read_initial_experiments, store_initial_experiments
from traceability_lab.main import create_app

CASES = [
    ("browser_file_sync", "test_file_sync"),
    ("browser_prompts", "test_prompts"),
    ("browser_first_server", "test_first_server"),
    ("browser_field_help", "test_help_requires_button_click"),
    ("browser_experiment_save", "test_save"),
    ("browser_experiment_json_save", "main"),
    ("browser_start_experiment", "test_saved_experiment_can_start"),
    ("browser_model_identity", "test_manifest"),
    ("browser_active_execution", "test_reconnect"),
    ("browser_reports", "test_reports"),
    ("browser_fact_average", "main"),
    ("browser_reviewer_prompt", "main"),
]


@pytest.mark.browser
@pytest.mark.parametrize("module_name,entry", CASES)
def test_browser_workflow(request, tmp_path, monkeypatch, module_name, entry):
    if not request.config.getoption("--browser"):
        pytest.skip("Use --browser to run isolated browser tests.")
    importlib.import_module("playwright.sync_api")
    settings = Settings(
        database=tmp_path / "lab.sqlite3",
        experiments_directory=PROJECT_ROOT / "experiments",
        runtime_metadata_default=False,
    )
    if module_name == "browser_file_sync":
        import shutil

        for folder in ("experiments", "prompts"):
            shutil.copytree(PROJECT_ROOT / folder, tmp_path / folder)
        settings = settings.model_copy(
            update={
                "experiments_directory": tmp_path / "experiments",
                "prompts_directory": tmp_path / "prompts",
            }
        )
    if module_name != "browser_first_server":
        from profile_presets import seed_test_profiles

        db = Database(settings.database)
        db.initialize()
        store_initial_experiments(db, read_initial_experiments(settings.experiments_directory))
        seed_test_profiles(db)
    app = create_app(settings)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert server.started, "The isolated browser server did not start."
            base_url = f"http://127.0.0.1:{port}"
            args = [base_url]
            if module_name == "browser_file_sync":
                args.append(tmp_path)
            if module_name in {"browser_reports", "browser_fact_average", "browser_reviewer_prompt"}:
                import json

                import httpx
                from test_quoted_facts import configure_calls, quoted_experiment

                from traceability_lab import semantic_evaluation

                configure_calls(monkeypatch, bad_quote=True)
                original = semantic_evaluation.generate

                def scored(*arguments):
                    response = original(*arguments)
                    message = response["choices"][0]["message"]
                    value = json.loads(message["content"])
                    judgment = value if "fact_index" in value else value["answer"]
                    judgment.update(correctness_score=100, confidence=100)
                    message["content"] = json.dumps(value)
                    return response

                monkeypatch.setattr(semantic_evaluation, "generate", scored)
                definition = quoted_experiment()
                definition["title"] = "00 Browser result regression"
                with httpx.Client(base_url=base_url, trust_env=False) as client:
                    saved = client.post("/api/experiment-definitions", json=definition)
                    assert saved.status_code == 201, saved.text
                    response = client.post(
                        "/api/experiment-definitions/run",
                        json={
                            "definition": definition,
                            "saved_id": saved.json()["id"],
                            "execution": {"repetitions": 2, "llm_profile_id": "test-generator"},
                        },
                    )
                    assert response.status_code == 201, response.text
                    args.append(response.json()["batch_id"])
            module = importlib.import_module(module_name)
            getattr(module, entry)(*args)
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            assert not thread.is_alive(), "The isolated browser server did not stop."
