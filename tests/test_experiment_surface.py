"""Only the experiment workflow is exposed by the production application."""

from fastapi.testclient import TestClient

from traceability_lab.main import create_app


def test_removed_workflows_are_not_public(settings):
    with TestClient(create_app(settings)) as client:
        routes = client.get("/openapi.json").json()["paths"]
        assert not any(
            path.startswith(("/api/documents", "/api/contexts", "/api/claims", "/api/questions"))
            for path in routes
        )
        assert not any(path.endswith(("/claims", "/reviews")) for path in routes)
        assert client.post("/api/questions", json={"question": "Example?"}).status_code == 404
        assert client.post("/api/context-preview", json={"question": "Example?"}).status_code == 422
        html = client.get("/").text
        for legacy in ["documents-dialog", "context-form", "embedding-profile", "choose-sources"]:
            assert f'id="{legacy}"' not in html


def test_assets_revalidate_without_manual_cache_tokens(settings):
    with TestClient(create_app(settings)) as client:
        index = client.get("/")
        assert "?v=" not in index.text
        assert index.headers["cache-control"] == "no-cache"
        asset = client.get("/static/app.js")
        assert "?v=" not in asset.text
        assert asset.headers["cache-control"] == "no-cache"
        assert (
            client.get("/static/app.js", headers={"If-None-Match": asset.headers["etag"]}).status_code == 304
        )


def test_public_execution_defaults_to_three_repetitions(settings):
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/experiment-definitions/run",
            json={
                "definition": {
                    "title": "Default repetitions",
                    "question": "What is known?",
                    "documents": [],
                    "output_mode": "facts_and_answer",
                }
            },
        )
        assert response.status_code == 201, response.text
        assert len(response.json()["run_ids"]) == 3
