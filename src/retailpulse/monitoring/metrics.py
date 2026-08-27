"""Monitoring metrics: rolling WAPE, bias, pinball, coverage, freshness.

Computed from the persisted forecast-vs-actual table. Warning/failure
thresholds come from ``configs/monitoring.yaml``; each alert carries a
metric, threshold, window, and a response playbook pointer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from retailpulse.config import get_config
from retailpulse.evaluation.metrics import bias, coverage, pinball, wape


@dataclass(frozen=True)
class Alert:
    """One triggered monitoring alert."""

    metric: str
    state: str  # "warning" | "failure"
    message: str
    playbook: str


def _as_float(value: float | list[float]) -> float:
    """Thresholds are scalar floats in monitoring.yaml (the coverage bands
    are lists, handled separately)."""
    if isinstance(value, list):
        raise TypeError(f"expected scalar threshold, got list: {value}")
    return float(value)


def rolling_metrics(
    forecast_actual: pd.DataFrame,
    *,
    window_days: int | None = None,
) -> dict[str, float]:
    """Rolling metrics over the configured window."""
    cfg = get_config().monitoring
    window = window_days or cfg.rolling_window_days
    df = forecast_actual.copy()
    if df.empty:
        return {
            "wape": float("nan"),
            "bias": float("nan"),
            "coverage": float("nan"),
            "pinball": float("nan"),
        }
    df["target_date"] = pd.to_datetime(df["target_date"])
    cutoff = df["target_date"].max() - pd.Timedelta(days=window)
    recent = df[df["target_date"] > cutoff]
    actual = recent["actual"].to_numpy(dtype=np.float64)
    q50 = recent["q50"].to_numpy(dtype=np.float64)
    valid = ~np.isnan(actual)
    if valid.sum() == 0:
        return {
            "wape": float("nan"),
            "bias": float("nan"),
            "coverage": float("nan"),
            "pinball": float("nan"),
        }
    actual, q50 = actual[valid], q50[valid]
    return {
        "wape": wape(actual, q50),
        "bias": bias(actual, q50),
        "coverage": coverage(
            actual,
            recent["q10"].to_numpy(dtype=np.float64)[valid],
            recent["q90"].to_numpy(dtype=np.float64)[valid],
        ),
        "pinball": np.mean(
            [
                pinball(actual, recent["q10"].to_numpy(dtype=np.float64)[valid], 0.1),
                pinball(actual, q50, 0.5),
                pinball(actual, recent["q90"].to_numpy(dtype=np.float64)[valid], 0.9),
            ]
        ),
    }


def check_alerts(
    metrics: dict[str, float],
    *,
    freshness_hours: float | None = None,
) -> list[Alert]:
    """Compare rolling metrics against warning/failure thresholds."""
    cfg = get_config().monitoring
    alerts: list[Alert] = []

    def state_of(value: float, warn: float, fail: float) -> str | None:
        if value >= fail:
            return "failure"
        if value >= warn:
            return "warning"
        return None

    m = cfg.model
    wape_v = metrics.get("wape")
    if wape_v is not None and not np.isnan(wape_v):
        warn = _as_float(m["wape_warn"])
        fail = _as_float(m["wape_fail"])
        s = state_of(wape_v, warn, fail)
        if s:
            alerts.append(Alert("wape", s, f"rolling WAPE {wape_v:.3f}", "retrain gate review"))

    bias_v = metrics.get("bias")
    if bias_v is not None and not np.isnan(bias_v):
        warn = _as_float(m["bias_warn"])
        fail = _as_float(m["bias_fail"])
        s = state_of(abs(bias_v), warn, fail)
        if s:
            alerts.append(Alert("bias", s, f"rolling bias {bias_v:.3f}", "bias audit"))

    cov = metrics.get("coverage")
    if cov is not None and not np.isnan(cov):
        band = m["coverage_fail_band"]
        if isinstance(band, list) and not (float(band[0]) <= cov <= float(band[1])):
            alerts.append(
                Alert(
                    "coverage",
                    "failure",
                    f"coverage {cov:.3f} outside {band}",
                    "calibration review",
                )
            )

    d = cfg.data
    if freshness_hours is not None:
        if freshness_hours >= float(d["freshness_fail_hours"]):
            alerts.append(
                Alert("freshness", "failure", f"data {freshness_hours:.1f}h old", "ingest retry")
            )
        elif freshness_hours >= float(d["freshness_warn_hours"]):
            alerts.append(
                Alert("freshness", "warning", f"data {freshness_hours:.1f}h old", "ingest retry")
            )

    return alerts
