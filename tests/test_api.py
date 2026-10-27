"""API-validatie en kleine lokale browsergrenzen."""

import pytest


def test_ui_assets_and_api_documentation_are_served(client):
    assert client.get("/").status_code == 200
    assert "AI Traceability Lab" in client.get("/").text
    assert 'id="mode"' not in client.get("/").text
    assert "Unrestricted" not in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200
    assert "https://cdn" not in client.get("/docs").text
    assert client.get("/api/health").json()["backend"] == "demo"
    assert "frame-ancestors 'none'" in client.get("/").headers["content-security-policy"]


@pytest.mark.parametrize(
    "payload",
    [
        {"question": " "},
        {"question": "x", "top_k": 0},
        {"question": "x", "version_ids": []},
        {"question": "x", "version_ids": ["unknown"]},
        {"question": "x", "mode": "unvalidated"},
        {"question": "x", "mode": "unrestricted"},
        {"question": "x", "mode": "strict"},
    ],
)
def test_invalid_request_never_creates_a_run(client, payload):
    assert client.post("/api/questions", json=payload).status_code == 422
    assert client.get("/api/runs").json() == []


def test_missing_records_and_external_urls(client):
    assert client.get("/api/runs/missing").status_code == 404
    assert client.get("/api/batches/missing").status_code == 404
    assert (
        client.post(
            "/api/questions", json={"question": "x"}, headers={"Origin": "https://example.com"}
        ).status_code
        == 201
    )
    assert client.get("/", headers={"Host": "external.example"}).status_code == 200
