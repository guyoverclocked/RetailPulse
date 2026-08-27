"""API integration tests against the FastAPI app."""

from __future__ import annotations

from fastapi.testclient import TestClient

from retailpulse.api.app import app

client = TestClient(app)


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["model_version"]


def test_forecasts_404_for_unknown_store() -> None:
    r = client.get("/v1/forecasts/999999")
    assert r.status_code == 404


def test_forecasts_invalid_horizon_rejected() -> None:
    r = client.get("/v1/forecasts/1?horizon=999")
    assert r.status_code == 422


def test_staffing_invalid_risk_mode_rejected() -> None:
    r = client.post(
        "/v1/staffing/optimize",
        json={"store_id": 1, "horizon_days": 7, "budget_hours": 100.0, "risk_mode": "p99"},
    )
    assert r.status_code == 422


def test_monitoring_summary_returns() -> None:
    r = client.get("/v1/monitoring/summary")
    assert r.status_code == 200
    assert "alerts" in r.json()
