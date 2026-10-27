"""Display names never route API calls; artifact claims have explicit trust boundaries."""

from copy import deepcopy

import httpx
import pytest

from traceability_lab.model_identity import capture_identity, recorded_identity
from traceability_lab.reproducibility import reproducibility_record

PROFILE = {
    "id": "generator",
    "name": "Descriptive model",
    "model": "test-generator-model",
    "url": "http://127.0.0.1:9000/v1",
}


def capture(profile=None):
    return capture_identity(profile or PROFILE, api_model_id="test-generator-model", url=PROFILE["url"])


def mock_props(monkeypatch, payloads):
    client = httpx.Client
    requests = []

    def handler(request):
        requests.append(request)
        payload = payloads.pop(0)
        return httpx.Response(503) if payload is None else httpx.Response(200, json=payload)

    monkeypatch.setattr(
        "traceability_lab.model_identity.httpx.Client",
        lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs),
    )
    return requests


def props(sha="a" * 64):
    return {
        "model_path": "/models/actual.gguf",
        "build_info": "b123-abcd",
        "chat_template": "exact template",
        "default_generation_settings": {"params": {"top_p": 0.9}},
        "traceability_artifact": {
            "sha256": sha,
            "api_model_id": "test-generator-model",
            "loaded": True,
            "deployment_id": "runtime-123",
            "engine": "llama.cpp",
        },
    }


@pytest.mark.parametrize(
    "manifest,status",
    [
        (None, "unknown"),
        ({"filename": "weights.gguf", "provenance": "deployment"}, "declared"),
        ({"sha256": "b" * 64, "provenance": "deployment"}, "declared"),
    ],
)
def test_missing_and_declared_artifacts_do_not_claim_verification(manifest, status):
    identity = capture({**PROFILE, "deployment_manifest": manifest})
    assert identity["display_name"] == "Descriptive model"
    assert identity["technical"]["api_model_id"] == "test-generator-model"
    assert identity["artifact"]["identity_status"] == status


def test_loaded_artifact_attestation_is_refreshed_not_cached(monkeypatch):
    requests = mock_props(monkeypatch, [props(), props("b" * 64), None])
    profile = {
        **PROFILE,
        "capture_server_metadata": True,
        "trust_server_identity": True,
        "deployment_manifest": {"sha256": "a" * 64, "provenance": "deployment"},
    }
    first, second, unavailable = (capture(profile) for _ in range(3))
    assert first["artifact"]["identity_status"] == second["artifact"]["identity_status"] == "verified"
    assert first["artifact"]["sha256"] == "a" * 64 and second["artifact"]["sha256"] == "b" * 64
    assert second["technical"]["declared_sha256_mismatch"] == "a" * 64
    assert unavailable["artifact"]["identity_status"] == "declared"
    assert "metadata_error" in unavailable["technical"]
    assert len(requests) == 3 and all(request.url.path == "/props" for request in requests)
    assert first["runtime"]["server_defaults"] == {"top_p": 0.9}
    assert first["runtime"]["chat_template"] == "exact template"
    assert first["runtime"]["revision"] == "b123-abcd"


@pytest.mark.parametrize(
    "change",
    [{"loaded": False}, {"api_model_id": "other"}, {"deployment_id": None}, {"sha256": "not-a-hash"}],
)
def test_invalid_or_wrong_model_attestation_never_verifies(monkeypatch, change):
    payload = props()
    payload["traceability_artifact"].update(change)
    mock_props(monkeypatch, [payload])
    assert (
        capture({**PROFILE, "capture_server_metadata": True, "trust_server_identity": True})["artifact"][
            "identity_status"
        ]
        != "verified"
    )


def test_untrusted_loaded_identity_remains_declared(monkeypatch):
    mock_props(monkeypatch, [props()])
    assert capture({**PROFILE, "capture_server_metadata": True})["artifact"]["identity_status"] == "declared"


def test_historical_names_use_only_recorded_profile_and_alias_stays_technical():
    model = {"id": "test-generator-model"}
    original = deepcopy(model)
    assert recorded_identity(model)["display_name"] == "Unknown model"
    identity = recorded_identity(model, PROFILE)
    assert identity["display_name"] == "Descriptive model"
    assert identity["technical"]["api_model_id"] == "test-generator-model"
    assert "id" not in identity and model == original
    assert capture({**PROFILE, "id": "another", "name": "Another model"})["display_name"] == "Another model"


def test_fingerprint_is_canonical_rename_invariant_and_sensitive_to_behavior():
    identity = capture({**PROFILE, "artifact_sha256": "a" * 64})
    run = {
        "snapshot": {
            "model": identity,
            "request_body": {
                "model": "test-generator-model",
                "temperature": 0.4,
                "messages": [{"role": "user", "content": "Question"}],
            },
            "source_tree_sha256": "code",
        },
        "context": [{"document_id": "d", "version": "1", "sha256": "source"}],
    }
    automatic = {
        "evaluator_version": "v1",
        "semantic_evaluation": {
            "model": capture({**PROFILE, "id": "review", "name": "Review model"}),
            "configuration": {"temperature": 0},
            "policies": {"fact": "policy"},
        },
    }
    before = reproducibility_record(run, automatic)
    renamed = deepcopy(run)
    renamed["snapshot"]["model"]["display_name"] = "New name"
    renamed_auto = deepcopy(automatic)
    renamed_auto["semantic_evaluation"]["model"]["display_name"] = "Renamed reviewer"
    assert (
        reproducibility_record(renamed, renamed_auto)["configuration_sha256"]
        == before["configuration_sha256"]
    )
    reordered = {"context": run["context"], "snapshot": dict(reversed(list(run["snapshot"].items())))}
    assert (
        reproducibility_record(reordered, automatic)["configuration_sha256"] == before["configuration_sha256"]
    )
    for section in ("artifact", "runtime"):
        changed = deepcopy(run)
        changed["snapshot"]["model"][section]["sha256" if section == "artifact" else "revision"] = "different"
        assert (
            reproducibility_record(changed, automatic)["configuration_sha256"]
            != before["configuration_sha256"]
        )
    assert "Descriptive model" not in str(before["inputs"])
    assert "generator.chat_template" in before["missing_provenance_fields"]


def test_profile_manifest_registration_validates_trust(client):
    data = {**PROFILE, "kind": "llm", "trust_server_identity": True}
    data.pop("id")
    assert client.post("/api/profiles", json=data).status_code == 422
    data.update(
        capture_server_metadata=True,
        deployment_manifest={"filename": "weights.gguf", "sha256": "a" * 64, "provenance": "deployment"},
    )
    saved = client.post("/api/profiles", json=data)
    assert saved.status_code == 201, saved.text
    assert saved.json()["deployment_manifest"]["sha256"] == "a" * 64


def test_generator_reviewer_exports_and_rename_snapshots(client, monkeypatch):
    from test_quoted_facts import configure_calls, quoted_experiment

    from traceability_lab.models import ProfileInput

    configure_calls(monkeypatch)

    def update(identity, **changes):
        old = client.get("/api/profiles").json()
        data = next(profile for profile in old if profile["id"] == identity)
        data = {key: value for key, value in data.items() if key in ProfileInput.model_fields}
        result = client.put(f"/api/profiles/{identity}", json={**data, **changes})
        assert result.status_code == 200, result.text

    update("test-generator", name="Small generator", artifact_sha256="a" * 64)
    update("test-reviewer", name="Large reviewer", model="test-generator-model")
    definition = quoted_experiment()
    definition["semantic_evaluator"]["llm_profile_id"] = "test-reviewer"

    def execute():
        result = client.post(
            "/api/experiment-definitions/run",
            json={
                "definition": definition,
                "execution": {"repetitions": 1, "llm_profile_id": "test-generator"},
            },
        )
        assert result.status_code == 201, result.text
        batch = client.get(f"/api/batches/{result.json()['batch_id']}").json()
        return client.get(f"/api/runs/{batch['runs'][0]['id']}/report").json()

    first = execute()
    assert first["models"]["generator"]["display_name"] == "Small generator"
    assert first["models"]["reviewer"]["display_name"] == "Large reviewer"
    assert first["models"]["generator"]["profile_id"] != first["models"]["reviewer"]["profile_id"]
    assert first["run"]["snapshot"]["request_body"]["model"] == "test-generator-model"
    assert all(
        call["request_body"]["model"] == "test-generator-model"
        for call in first["evaluation"]["automatic"]["semantic_evaluation"]["calls"]
    )
    assert "id" not in first["models"]["generator"]
    update("test-generator", name="Renamed generator")
    update("test-reviewer", name="Renamed reviewer")
    second = execute()
    assert second["models"]["generator"]["display_name"] == "Renamed generator"
    assert (
        first["reproducibility"]["configuration_sha256"] == second["reproducibility"]["configuration_sha256"]
    )
    historical = client.get(f"/api/runs/{first['run']['id']}/report").json()
    assert historical["run"] == first["run"]
    assert historical["models"] == first["models"]
    update("test-generator", artifact_sha256="b" * 64)
    third = execute()
    assert third["models"]["generator"]["display_name"] == second["models"]["generator"]["display_name"]
    assert (
        third["reproducibility"]["configuration_sha256"] != second["reproducibility"]["configuration_sha256"]
    )
    listed = client.get("/api/runs").json()
    assert next(run for run in listed if run["id"] == first["run"]["id"])["model_id"] == "Small generator"
