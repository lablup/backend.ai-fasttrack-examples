"""The API's auth posture, checked at the request path rather than at boot.

`require_auth_configured()` refuses to start without a key, but it runs in
`main()` — so anything that serves the ASGI object directly (`uvicorn
docs_rag.server:app`, a gunicorn worker, an overridden container command) skips
it. These tests pin the behaviour of the dependency itself, which is the only
guard that path still has.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import docs_rag.server as server
from docs_rag.settings import Settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Build a client with a chosen API key and no indices to load."""

    def _make(api_key: str, allow_unauthenticated: bool = False):
        monkeypatch.setenv("DOCSRAG_INDICES", str(tmp_path / "empty"))
        if allow_unauthenticated:
            monkeypatch.setenv("ALLOW_UNAUTHENTICATED", "1")
        else:
            monkeypatch.delenv("ALLOW_UNAUTHENTICATED", raising=False)
        # Bypass the cached get_settings(): the lifespan builds real settings,
        # and we only care about the api_key the dependency reads.
        monkeypatch.setattr(
            server, "get_settings", lambda: Settings(api_key=api_key, openai_api_key="x")
        )
        return TestClient(server.app)

    return _make


def test_a_valid_token_is_accepted(client):
    with client("s3cret") as c:
        assert c.get("/v1/models", headers={"Authorization": "Bearer s3cret"}).status_code == 200


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "s3cret"}],
    ids=["no header", "wrong token", "malformed scheme"],
)
def test_a_bad_or_missing_token_is_rejected(client, headers):
    with client("s3cret") as c:
        assert c.get("/v1/models", headers=headers).status_code == 401


def test_an_unconfigured_server_refuses_to_serve(client):
    """Fails closed at the request path, not only at the entrypoint.

    Before this, an empty API_KEY made `verify_token` return successfully, so a
    boot that skipped main() served the whole corpus to anyone — and /health
    still reported 200, so nothing distinguished it from a healthy service.
    """
    with client("") as c:
        response = c.get("/v1/models")
        assert response.status_code == 503
        assert "no API_KEY" in response.json()["detail"]


def test_serving_open_is_still_possible_when_asked_for_explicitly(client):
    """ALLOW_UNAUTHENTICATED is a documented, deliberate opt-out."""
    with client("", allow_unauthenticated=True) as c:
        assert c.get("/v1/models").status_code == 200


def test_health_never_requires_a_token(client):
    """The model-service probe has no credentials to offer."""
    with client("s3cret") as c:
        assert c.get("/health").status_code == 200
