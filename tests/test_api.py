"""Tests for the FastAPI serving layer."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked engine."""
    from pravaha.serving.app import create_app
    app = create_app()
    app.state.engine = MagicMock()
    app.state.engine.get_stats.return_value = {"total_requests": 0}
    app.state.engine.is_ready = True
    app.state.engine.tokenizer = MagicMock()
    app.state.engine.tokenizer.encode.return_value = [1, 2, 3]
    app.state.engine.config = MagicMock()
    app.state.engine.config.model.model_path = "mock-model"
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status(self, client: TestClient) -> None:
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data


class TestModelsEndpoint:
    def test_models_returns_list(self, client: TestClient) -> None:
        resp = client.get("/v1/models")
        assert resp.status_code == 200


class TestSwarmEndpoints:
    def test_list_agents(self, client: TestClient) -> None:
        resp = client.get("/v1/swarm/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert "agents" in data

    def test_list_pipelines(self, client: TestClient) -> None:
        resp = client.get("/v1/swarm/pipelines")
        assert resp.status_code == 200
        data = resp.json()
        assert "pipelines" in data


class TestMiddleware:
    def test_request_id_header(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert "x-request-id" in resp.headers

    def test_process_time_header(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert "x-process-time" in resp.headers
