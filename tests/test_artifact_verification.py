"""Real tiny GGUF fixtures and mocked Linux/server observations; never live inference."""

import hashlib
import json
import os
from copy import deepcopy

import pytest

from traceability_lab import artifact_files
from traceability_lab.artifact_files import file_identity, hash_artifact, process_binding, recheck_artifact
from traceability_lab.model_identity import capture_identity
from traceability_lab.models import DeploymentManifest
from traceability_lab.reproducibility import reproducibility_record


@pytest.fixture
def artifact(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF" + bytes(range(256)) * 100)
    return path


def test_complete_artifact_hash_and_size(artifact):
    receipt = hash_artifact(str(artifact))
    assert receipt["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert receipt["size_bytes"] == artifact.stat().st_size
    assert receipt["filename"] == "model.gguf"
    assert receipt["file_stat"] == file_identity(artifact.stat())


def test_not_a_gguf_is_rejected(tmp_path):
    path = tmp_path / "not.gguf"
    path.write_bytes(b"not a GGUF")
    with pytest.raises(ValueError, match="regular GGUF"):
        hash_artifact(str(path))


def test_manifest_import_and_invalid_sha():
    manifest = DeploymentManifest.model_validate(
        {
            "artifact": {"filename": "actual.gguf", "sha256": "a" * 64, "size_bytes": 4096},
            "runtime": {"engine": "llama.cpp", "revision": "b123"},
            "verification": {"verified_at": "deployment-time"},
        }
    )
    assert manifest.sha256 == "a" * 64 and manifest.size_bytes == 4096
    assert manifest.engine_revision == "b123"
    with pytest.raises(ValueError):
        DeploymentManifest.model_validate({"sha256": "not-a-sha", "provenance": "test"})
    with pytest.raises(ValueError, match="Conflicting"):
        DeploymentManifest.model_validate({"sha256": "b" * 64, "artifact": {"sha256": "a" * 64}})


def mock_linux(tmp_path, artifact):
    proc = tmp_path / "proc"
    root = proc / "123"
    (root / "ns").mkdir(parents=True)
    (proc / "self/ns").mkdir(parents=True)
    (root / "ns/net").symlink_to("net:[123]")
    (proc / "self/ns/net").symlink_to("net:[123]")
    (root / "fd").mkdir(parents=True)
    (root / "net").mkdir()
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "sys/kernel/random/boot_id").write_text("boot-1")
    (root / "stat").write_text("123 (llama server) " + " ".join(["S"] + ["0"] * 18 + ["100"]))
    value = artifact.stat()
    (root / "maps").write_text(
        f"1000-2000 r--p 0000 {os.major(value.st_dev):02x}:{os.minor(value.st_dev):02x} {value.st_ino} {artifact}\n"
    )
    (root / "fd/5").symlink_to("socket:[77]")
    (root / "exe").symlink_to("/usr/bin/llama-server")
    (root / "cmdline").write_bytes(
        f"llama-server\0--model\0{artifact}\0--api-key\0secret-do-not-record\0-ngl\099\0".encode()
    )
    (root / "net/tcp").write_text("header\n0: 0100007F:1F40 00000000:0000 0A 0:0 0:0 00 1000 0 77\n")
    return proc


def test_process_mapping_and_listener_not_arguments_alone(tmp_path, artifact):
    proc = mock_linux(tmp_path, artifact)
    binding = process_binding(str(artifact), 123, "http://127.0.0.1:8000/v1", proc_root=proc)
    assert binding["start_time_ticks"] == "100"
    assert "secret-do-not-record" not in json.dumps(binding)
    (proc / "123/maps").write_text("")
    with pytest.raises(ValueError, match="not mapped"):
        process_binding(str(artifact), 123, "http://127.0.0.1:8000/v1", proc_root=proc)


def test_wrong_listener_cannot_verify(tmp_path, artifact):
    proc = mock_linux(tmp_path, artifact)
    (proc / "123/net/tcp").write_text("header\n0: FD02A8C0:1F40 00000000:0000 0A 0:0 0:0 00 1000 0 77\n")
    with pytest.raises(ValueError, match="listener"):
        process_binding(str(artifact), 123, "http://127.0.0.1:8000/v1", proc_root=proc)


def test_local_verification_restart_and_file_replacement_without_hashing(tmp_path, artifact, monkeypatch):
    proc = mock_linux(tmp_path, artifact)
    original = process_binding
    monkeypatch.setattr(
        artifact_files, "process_binding", lambda path, pid, url: original(path, pid, url, proc_root=proc)
    )
    receipt = hash_artifact(str(artifact))
    receipt.update(
        url="http://127.0.0.1:8000/v1",
        api_model_id="alias",
        process=artifact_files.process_binding(str(artifact), 123, "http://127.0.0.1:8000/v1"),
    )
    profile = {"url": receipt["url"], "model": "alias"}
    props = {"model_path": str(artifact)}
    monkeypatch.setattr(artifact_files, "hash_artifact", lambda *a: pytest.fail("Hashing during inference"))
    assert recheck_artifact(receipt, profile, props)["status"] == "loaded_verified"
    (proc / "123/stat").write_text("123 (llama server) " + " ".join(["S"] + ["0"] * 18 + ["101"]))
    assert recheck_artifact(receipt, profile, props)["status"] == "stale"
    artifact.write_bytes(b"GGUF replacement")
    assert "file changed" in recheck_artifact(receipt, profile, props)["reason"]


def test_explicit_profile_registration_and_rename_preserves_receipt(client, artifact):
    response = client.post("/api/profiles/test-generator/verify-artifact", json={"path": str(artifact)})
    assert response.status_code == 200, response.text
    profile = response.json()
    assert profile["deployment_manifest"]["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert profile["local_artifact_verification"]["size_bytes"] == artifact.stat().st_size
    from traceability_lab.models import ProfileInput

    data = {key: value for key, value in profile.items() if key in ProfileInput.model_fields}
    renamed = client.put("/api/profiles/test-generator", json={**data, "name": "Renamed"}).json()
    assert renamed["local_artifact_verification"] == profile["local_artifact_verification"]
    changed = client.put("/api/profiles/test-generator", json={**data, "model": "different"}).json()
    assert "local_artifact_verification" not in changed


def test_imported_verification_is_declared_not_a_local_receipt(artifact):
    manifest = DeploymentManifest.model_validate(
        {
            "artifact": {
                "filename": artifact.name,
                "sha256": "a" * 64,
                "size_bytes": artifact.stat().st_size,
            },
            "verification": {"status": "verified", "path": str(artifact)},
        }
    ).model_dump()
    identity = capture_identity(
        {"deployment_manifest": manifest}, api_model_id="alias", url="http://127.0.0.1:8000/v1"
    )
    assert identity["artifact"]["identity_status"] == "declared"


def test_fingerprint_v2_distinguishes_verified_from_declared_but_not_names():
    model = capture_identity(
        {"name": "One", "artifact_sha256": "a" * 64}, api_model_id="alias", url="http://127.0.0.1:8000/v1"
    )
    run = {"snapshot": {"model": model}, "context": []}
    first = reproducibility_record(run, {})
    assert first["version"] == "configuration-fingerprint-v2"
    renamed = deepcopy(run)
    renamed["snapshot"]["model"]["display_name"] = "Two"
    assert reproducibility_record(renamed, {})["configuration_sha256"] == first["configuration_sha256"]
    renamed["snapshot"]["model"]["artifact"]["identity_status"] = "verified"
    assert reproducibility_record(renamed, {})["configuration_sha256"] != first["configuration_sha256"]


def mock_server(monkeypatch, response):
    import httpx

    original = httpx.Client
    monkeypatch.setattr(
        "httpx.Client",
        lambda **kwargs: original(transport=httpx.MockTransport(lambda request: response(request)), **kwargs),
    )


def test_auto_runtime_capture_and_local_loaded_identity_without_hashing(tmp_path, artifact, monkeypatch):
    import httpx

    proc = mock_linux(tmp_path, artifact)
    original = process_binding
    monkeypatch.setattr(
        artifact_files, "process_binding", lambda path, pid, url: original(path, pid, url, proc_root=proc)
    )
    receipt = hash_artifact(str(artifact))
    receipt.update(
        url="http://127.0.0.1:8000/v1",
        api_model_id="alias",
        process=artifact_files.process_binding(str(artifact), 123, "http://127.0.0.1:8000/v1"),
    )
    profile = {
        "name": "Frozen name",
        "model": "alias",
        "url": receipt["url"],
        "local_artifact_verification": receipt,
        "deployment_manifest": {
            "sha256": receipt["sha256"],
            "filename": receipt["filename"],
            "size_bytes": receipt["size_bytes"],
        },
        "capture_server_metadata": None,
    }
    props = {
        "model_path": str(artifact),
        "model_ftype": "Q4_K - Medium",
        "build_info": "b123-abc",
        "chat_template": "exact template",
        "total_slots": 2,
        "default_generation_settings": {"n_ctx": 4096, "params": {"top_p": 0.9}},
    }
    mock_server(monkeypatch, lambda request: httpx.Response(200, json=props, headers={"server": "llama.cpp"}))
    monkeypatch.setattr(
        artifact_files, "hash_artifact", lambda *args: pytest.fail("Hashing during inference")
    )
    identity = capture_identity(profile, api_model_id="alias", url=receipt["url"], capture_default=True)
    assert identity["artifact"]["identity_status"] == "verified"
    assert identity["artifact"]["sha256"] == receipt["sha256"]
    assert identity["runtime"]["engine"] == "llama.cpp" and identity["runtime"]["revision"] == "b123-abc"
    assert identity["runtime"]["server_configuration"]["n_ctx"] == 4096
    assert identity["runtime"]["server_configuration"]["process_arguments"]["-ngl"] == "99"
    assert identity["artifact"]["field_sources"]["sha256"] == "locally_verified"
    assert identity["runtime"]["field_sources"]["engine"] == "server_observed"
    assert identity["runtime"]["gpu_offloading"] is None  # Requested layers are not actual GPU execution.
    assert identity["runtime"]["chat_template_sha256"] == hashlib.sha256(b"exact template").hexdigest()


def test_unavailable_provenance_does_not_fail_inference(client, monkeypatch):
    import httpx
    from test_quoted_facts import configure_calls, execute

    from traceability_lab.models import ProfileInput

    configure_calls(monkeypatch)
    profile = next(item for item in client.get("/api/profiles").json() if item["id"] == "test-generator")
    data = {key: value for key, value in profile.items() if key in ProfileInput.model_fields}
    assert (
        client.put("/api/profiles/test-generator", json={**data, "capture_server_metadata": True}).status_code
        == 200
    )
    mock_server(monkeypatch, lambda request: httpx.Response(503))
    batch, observation = execute(client)
    assert batch["counts"]["completed"] == 1 and observation["status"] == "completed"
    run = client.get(f"/api/runs/{batch['runs'][0]['id']}").json()
    assert "metadata_error" in run["snapshot"]["model"]["technical"]


def test_both_role_snapshots_are_captured_before_generation(client, monkeypatch):
    from test_quoted_facts import configure_calls, execute

    import traceability_lab.pipeline as pipeline

    configure_calls(monkeypatch)
    clock = {"engine": "before-generation"}
    original_generate = pipeline.generate
    original_capture = capture_identity

    def capture(profile, **kwargs):
        value = original_capture(profile, **kwargs)
        value["runtime"]["engine"] = clock["engine"]
        return value

    def generate(*args):
        clock["engine"] = "after-generation"
        return original_generate(*args)

    monkeypatch.setattr("traceability_lab.pipeline.capture_identity", capture)
    monkeypatch.setattr("traceability_lab.semantic_evaluation.capture_identity", capture)
    monkeypatch.setattr("traceability_lab.pipeline.generate", generate)
    batch, observation = execute(client)
    run = client.get(f"/api/runs/{batch['runs'][0]['id']}").json()
    assert run["snapshot"]["model"]["runtime"]["engine"] == "before-generation"
    assert observation["model"]["runtime"]["engine"] == "before-generation"
    assert run["snapshot"]["experiment"]["semantic_evaluator_snapshot"]["model"] == observation["model"]


def test_manifest_cli_runs_without_application_dependencies(artifact):
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-S", artifact_files.__file__, str(artifact), "--revision", "b123"],
        capture_output=True,
        text=True,
        check=True,
    )
    manifest = json.loads(result.stdout)
    assert manifest["artifact"]["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert manifest["runtime"]["revision"] == "b123"
    assert DeploymentManifest.model_validate(manifest).size_bytes == artifact.stat().st_size


def test_cancel_during_metadata_resolution_is_cancelled_not_failed(client, db, settings, monkeypatch):
    from threading import Event

    from test_defined_experiments import definition

    from traceability_lab.cancellation import RunCancelled
    from traceability_lab.defined_experiments import prepare_definition_runs
    from traceability_lab.models import DefinedExperimentRun

    cancel = Event()

    def cancelled(*args, **kwargs):
        cancel.set()
        raise RunCancelled("Cancelled metadata read")

    monkeypatch.setattr("traceability_lab.pipeline.capture_identity", cancelled)
    monkeypatch.setattr(
        "traceability_lab.pipeline.generate", lambda *args: pytest.fail("Generation after Cancel")
    )
    events = list(
        prepare_definition_runs(
            db,
            settings,
            client.app.state.embedder,
            client.app.state.tracing,
            DefinedExperimentRun(definition=definition(client)),
            cancel_event=cancel,
        )
    )
    assert events[-1]["type"] == "cancelled" and events[-1]["run_ids"] == []
