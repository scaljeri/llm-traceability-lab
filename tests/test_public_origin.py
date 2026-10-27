"""Incoming hostnames and origins are unrestricted by explicit project choice."""

import pytest


@pytest.mark.parametrize("host", ["example.trycloudflare.com", "demo.example.org", "10.0.0.10:8000"])
def test_external_hosts_and_writes_are_accepted(client, host):
    headers = {"host": host, "origin": "https://another.example", "sec-fetch-site": "cross-site"}
    assert client.get("/", headers=headers).status_code == 200
    response = client.post(
        "/api/profiles",
        headers=headers,
        json={"name": "External access", "kind": "llm", "url": "http://127.0.0.1:9000/v1", "model": "test"},
    )
    assert response.status_code == 201
    assert response.headers["x-content-type-options"] == "nosniff"
