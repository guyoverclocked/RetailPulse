"""Monitoring metrics and persistence tests."""

from __future__ import annotations

import pandas as pd

from retailpulse.monitoring.metrics import check_alerts, rolling_metrics


def test_rolling_metrics_on_small_frame() -> None:
    df = pd.DataFrame(
        {
            "origin": ["2020-01-01"] * 4,
            "target_date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]),
            "store_id": [1, 1, 2, 2],
            "model_version": ["v0"] * 4,
            "run_id": ["r1"] * 4,
            "q10": [80.0, 90.0, 180.0, 190.0],
            "q50": [100.0, 110.0, 200.0, 210.0],
            "q90": [120.0, 130.0, 220.0, 230.0],
            "actual": [105.0, 108.0, 205.0, 215.0],
        }
    )
    m = rolling_metrics(df, window_days=30)
    assert 0 < m["wape"] < 0.2
    # Bias is in raw sales units; -3.25 on ~150 sales is a healthy band.
    assert abs(m["bias"]) < 5.0
    assert 0.5 <= m["coverage"] <= 1.0


def test_alerts_fire_on_bad_wape() -> None:
    alerts = check_alerts({"wape": 0.9, "bias": 0.0, "coverage": 0.8}, freshness_hours=1.0)
    states = {a.metric: a.state for a in alerts}
    assert states.get("wape") == "failure"


def test_alerts_fire_on_freshness() -> None:
    alerts = check_alerts({"wape": 0.1, "bias": 0.0, "coverage": 0.8}, freshness_hours=100.0)
    states = {a.metric: a.state for a in alerts}
    assert states.get("freshness") == "failure"


def test_no_alerts_when_healthy() -> None:
    alerts = check_alerts({"wape": 0.1, "bias": 0.01, "coverage": 0.8}, freshness_hours=1.0)
    assert alerts == []
