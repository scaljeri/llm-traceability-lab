"""Serverprofielen en echte HTTP-embeddingcontracten zonder een GPU te vereisen."""

import json
import math

import httpx
import pytest

from traceability_lab.config import Settings
from traceability_lab.embeddings import Embedder


@pytest.fixture
def fake_servers(monkeypatch):
    original_client = httpx.Client
    calls = []

    def handler(request):
        body = json.loads(request.content) if request.content else None
        calls.append((str(request.url), body))
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "test-generator-model"}, {"id": "bge-m3"}]})
        if request.url.path.endswith("/embeddings"):
            # Bewust niet genormaliseerd: de client moet daar zelf voor zorgen.
            return httpx.Response(
                200,
                json={
                    "model": body["model"],
                    "data": [{"index": 0, "embedding": [3.0, 4.0] + [0.0] * 1022}],
                    "usage": {"total_tokens": 8},
                },
            )
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                200,
                json={
                    "model": body["model"],
                    "choices": [
                        {
                            "message": {"content": "Project Orion is gestart op 12 maart 2024."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"completion_tokens": 12, "prompt_tokens": 50},
                },
            )
        return httpx.Response(404)

    monkeypatch.setattr(
        "httpx.Client", lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs)
    )
    return calls


def test_presets_and_editable_profiles_persist_across_restarts(client, db, settings):
    profiles = client.get("/api/profiles").json()
    assert {p["id"] for p in profiles} == {"test-generator", "test-embedding", "test-reviewer"}
    llm = next(p for p in profiles if p["id"] == "test-generator")
    assert llm["model"] == "test-generator-model"
    data = {
        "name": "Jetson",
        "kind": "llm",
        "url": "http://10.0.0.10:8000/v1",
        "model": "small-model",
        "hardware": "Jetson",
    }
    created = client.post("/api/profiles", json=data)
    assert created.status_code == 201, created.text
    identity = created.json()["id"]
    assert client.put(f"/api/profiles/{identity}", json={**data, "model": "new-model"}).status_code == 200
    assert db.one("SELECT COUNT(*) n FROM model_profiles")["n"] == 4
    from fastapi.testclient import TestClient
    from legacy_api import legacy_app as create_app

    with TestClient(create_app(settings)) as restarted:
        assert (
            next(p for p in restarted.get("/api/profiles").json() if p["id"] == identity)["model"]
            == "new-model"
        )


@pytest.mark.parametrize(
    "change",
    [
        {"url": "https://public.example/v1"},
        {"url": "http://127.0.0.1:8000"},
        {"url": "http://user:secret@127.0.0.1/v1"},
        {"model": " "},
        {"kind": "embedding", "dimensions": None},
        {"dimensions": 1024},
    ],
)
def test_invalid_profiles_are_rejected(client, change):
    data = {"name": "Test", "kind": "llm", "url": "http://127.0.0.1:9000/v1", "model": "model"}
    assert client.post("/api/profiles", json={**data, **change}).status_code == 422


def test_selected_servers_drive_pipeline_and_old_snapshots_survive_profile_edits(client, fake_servers):
    selection = {"llm_profile_id": "test-generator", "embedding_profile_id": "test-embedding"}
    batch = client.post(
        "/test-scenarios/complete/run", json={**selection, "max_tokens": 1024, "seed": 7}
    ).json()
    run_id = batch["run_ids"][0]
    run = client.get(f"/api/runs/{run_id}").json()
    assert run["status"] == "completed", run["error"]
    assert run["backend"] == "local"
    assert run["snapshot"]["model"]["id"] == "test-generator-model"
    assert run["snapshot"]["profiles"]["llm"]["hardware"] == "GPU-server"
    assert run["snapshot"]["embedding"]["id"] == "bge-m3"
    assert run["snapshot"]["embedding"]["precision"] == "FP16"
    assert len(run["snapshot"]["embedding_observations"]) == 6
    vector = run["snapshot"]["query_vector"]
    assert math.isclose(math.hypot(*vector), 1)
    assert vector[:2] == [0.6, 0.8]
    chat = next(body for url, body in fake_servers if url.endswith("/chat/completions"))
    assert chat["model"] == "test-generator-model" and chat["seed"] == 7 and chat["max_tokens"] == 1024
    assert chat["chat_template_kwargs"] == {"enable_thinking": False}
    assert all(body["model"] == "bge-m3" for url, body in fake_servers if url.endswith("/embeddings"))
    profile = run["snapshot"]["profiles"]["llm"]
    updated = {key: value for key, value in profile.items() if key not in {"id", "created_at", "updated_at"}}
    updated.update(model="new-alias", name="Changed Jetson", hardware="Jetson")
    assert client.put("/api/profiles/test-generator", json=updated).status_code == 200
    assert client.get(f"/api/runs/{run_id}").json()["snapshot"] == run["snapshot"]
    history = client.get("/api/runs").json()[0]
    assert history["llm_name"] == "Test generator"
    assert history["output_tokens"] == 12 and history["llm_call_ms"] >= 0


def test_mixed_or_wrong_profile_roles_never_start_a_run(client):
    for data in [
        {"llm_profile_id": "test-generator"},
        {"llm_profile_id": "test-embedding", "embedding_profile_id": "test-generator"},
    ]:
        assert client.post("/api/questions", json={"question": "Test?", **data}).status_code == 422
    assert client.get("/api/runs").json() == []


def test_model_check_reports_availability_without_inference(client, fake_servers):
    check = client.post("/api/profiles/test-generator/check").json()
    assert check["reachable"] and check["model_available"]
    assert not check["inference_checked"]
    assert all(url.endswith("/models") for url, _ in fake_servers)


def test_batch_keeps_original_profiles_when_configuration_changes_mid_batch(
    client, db, fake_servers, monkeypatch
):
    from traceability_lab import pipeline
    from traceability_lab.models import ProfileInput
    from traceability_lab.profiles import read_profile, save_profile

    original = pipeline.generate
    calls = 0

    def generate_and_edit(*args):
        nonlocal calls
        response = original(*args)
        calls += 1
        if calls == 1:
            profile = read_profile(db, "test-generator")
            data = {
                key: value for key, value in profile.items() if key not in {"id", "created_at", "updated_at"}
            }
            save_profile(db, ProfileInput(**{**data, "model": "changed-mid-batch"}), "test-generator")
        return response

    monkeypatch.setattr(pipeline, "generate", generate_and_edit)
    batch = client.post(
        "/test-scenarios/complete/run",
        json={"llm_profile_id": "test-generator", "embedding_profile_id": "test-embedding", "repetitions": 2},
    ).json()
    for identity in batch["run_ids"]:
        run = client.get(f"/api/runs/{identity}").json()
        assert run["status"] == "completed"
        assert run["snapshot"]["profiles"]["llm"]["model"] == "test-generator-model"
        assert run["snapshot"]["request_body"]["model"] == "test-generator-model"
    assert read_profile(db, "test-generator")["model"] == "changed-mid-batch"


def test_schema_one_migration_preserves_existing_sources(client, db, settings):
    from fastapi.testclient import TestClient
    from legacy_api import legacy_app as create_app

    original_sources = client.get("/api/documents").json()
    # Een databaseschema-1-situatie namaken in uitsluitend de tijdelijke testdatabase.
    with db.connect() as connection:
        connection.execute("DROP TABLE model_profiles")
        connection.execute("PRAGMA user_version=1")
    with TestClient(create_app(settings)) as restarted:
        assert restarted.get("/api/documents").json() == original_sources
        assert restarted.get("/api/profiles").json() == []
    with db.connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 8


def test_remote_embedding_cache_is_reused_only_with_explicit_version(client, fake_servers):
    url = "/api/profiles/test-embedding"
    profile = next(p for p in client.get("/api/profiles").json() if p["id"] == "test-embedding")
    profile = {key: value for key, value in profile.items() if key not in {"id", "created_at", "updated_at"}}
    profile.update(revision="pinned-test-revision", query_prefix="query: ", document_prefix="passage: ")
    assert client.put(url, json=profile).status_code == 200
    batch = client.post(
        "/test-scenarios/complete/run",
        json={"llm_profile_id": "test-generator", "embedding_profile_id": "test-embedding", "repetitions": 2},
    ).json()
    assert len(batch["run_ids"]) == 2
    embedding_calls = [body for url, body in fake_servers if url.endswith("/embeddings")]
    assert len(embedding_calls) == 7  # vijf bronchunks één keer, twee queryvectoren
    assert sum(body["input"].startswith("query: ") for body in embedding_calls) == 2
    assert sum(body["input"].startswith("passage: ") for body in embedding_calls) == 5
    unknown = Settings(backend="local", embedding_url="http://127.0.0.1:9000/v1")
    assert Embedder(unknown).key() != Embedder(unknown).key()


@pytest.mark.parametrize("vector", [[0.0] * 1024, [1.0], [True] * 1024])
def test_invalid_api_vectors_are_rejected(monkeypatch, vector):
    original = httpx.Client
    monkeypatch.setattr(
        "httpx.Client",
        lambda **kwargs: original(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json={"data": [{"embedding": vector}]})
            ),
            **kwargs,
        ),
    )
    embedder = Embedder(Settings(backend="local", embedding_url="http://127.0.0.1:9000/v1"))
    with pytest.raises(ValueError):
        embedder.encode("Een tekst")


def test_first_start_has_no_preconfigured_servers(settings):
    from fastapi.testclient import TestClient
    from legacy_api import legacy_app as create_app

    with TestClient(create_app(settings)) as fresh:
        assert fresh.get("/api/profiles").json() == []
        saved = fresh.post(
            "/api/profiles",
            json={
                "name": "My reviewer",
                "kind": "llm",
                "url": "http://localhost:8080/v1",
                "model": "user-model",
            },
        )
        assert saved.status_code == 201
    with TestClient(create_app(settings)) as restarted:
        profiles = restarted.get("/api/profiles").json()
        assert len(profiles) == 1
        assert profiles[0]["name"] == "My reviewer"


def test_load_models_before_saving_profile(client, fake_servers, db):
    before = db.one("SELECT COUNT(*) n FROM model_profiles")["n"]
    response = client.post(
        "/api/profile-models",
        json={"name": "Discovery", "kind": "llm", "url": "http://localhost:8080/v1", "model": "discovery"},
    )
    assert response.status_code == 200
    assert response.json()["models"] == ["test-generator-model", "bge-m3"]
    assert fake_servers == [("http://localhost:8080/v1/models", None)]
    assert db.one("SELECT COUNT(*) n FROM model_profiles")["n"] == before


def test_delete_profile_preserves_recorded_runs(client, completed, settings):
    response = client.delete("/api/profiles/test-generator")
    assert response.status_code == 200
    assert not any(p["id"] == "test-generator" for p in client.get("/api/profiles").json())
    assert client.get(f"/api/runs/{completed['id']}").json() == completed
    assert client.delete("/api/profiles/test-generator").status_code == 404


def test_profile_output_token_limit_persists_and_validates(client):
    data = {
        "name": "Configurable output",
        "kind": "llm",
        "url": "http://localhost:8000/v1",
        "model": "test-model",
        "max_tokens": 6144,
    }
    response = client.post("/api/profiles", json=data)
    assert response.status_code == 201, response.text
    profile = response.json()
    assert profile["max_tokens"] == 6144
    profiles = client.get("/api/profiles").json()
    assert next(item for item in profiles if item["id"] == profile["id"])["max_tokens"] == 6144
    data["max_tokens"] = 2048
    assert client.put(f"/api/profiles/{profile['id']}", json=data).json()["max_tokens"] == 2048
    for invalid in (0, 8193, 1.5):
        assert (
            client.put(f"/api/profiles/{profile['id']}", json={**data, "max_tokens": invalid}).status_code
            == 422
        )
