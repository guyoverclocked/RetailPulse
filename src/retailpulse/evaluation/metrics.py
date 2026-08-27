"""Forecast metrics — the canonical scorecard.

Formulas (matching ``vault/03 Learn/Forecast Metrics.md``):

- WAPE = Σ|y - ŷ| / Σ|y|
- MAE  = (1/N) Σ|y - ŷ|
- Bias = (1/N) Σ(ŷ - y)      (positive = over-forecasting)
- MASE = MAE_model / MAE_seasonal_naive (same folds, naive errors from its own fit)
- Pinball L_q = q·(y-ŷ_q) if y ≥ ŷ_q else (1-q)·(ŷ_q - y)
- Coverage = (1/N) Σ I(ŷ_10 ≤ y ≤ ŷ_90)
- Width    = (1/N) Σ(ŷ_90 - ŷ_10)
- Skill vs seasonal naive = 1 - WAPE_model / WAPE_naive

Every function takes numpy arrays and ignores NaN pairs consistently.
Closed-store rows must be handled by callers via deterministic zeros: both
forecast and actual are zero there, so they contribute nothing to errors but
do count toward N (matching the "closed stores are structural zeros" rule).
"""

from __future__ import annotations

import numpy as np

Q_LOW = 0.10
Q_MID = 0.50
Q_HIGH = 0.90


def _aligned(actual: np.ndarray, forecast: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=np.float64)
    f = np.asarray(forecast, dtype=np.float64)
    if a.shape != f.shape:
        raise ValueError(f"shape mismatch: actual {a.shape} vs forecast {f.shape}")
    mask = ~(np.isnan(a) | np.isnan(f))
    return a[mask], f[mask]


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    a, f = _aligned(actual, forecast)
    denom = np.abs(a).sum()
    if denom == 0:
        return 0.0 if np.abs(f).sum() == 0 else float("inf")
    return float(np.abs(a - f).sum() / denom)


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    a, f = _aligned(actual, forecast)
    return float(np.abs(a - f).mean())


def bias(actual: np.ndarray, forecast: np.ndarray) -> float:
    a, f = _aligned(actual, forecast)
    return float((f - a).mean())


def mase(actual: np.ndarray, forecast: np.ndarray, naive_forecast: np.ndarray) -> float:
    """MASE = MAE(model) / MAE(seasonal naive) on identical positions."""
    a, f = _aligned(actual, forecast)
    _, nf = _aligned(actual, naive_forecast)
    denom = np.abs(a - nf).mean()
    if denom == 0:
        return 0.0 if np.abs(a - f).mean() == 0 else float("inf")
    return float(np.abs(a - f).mean() / denom)


def pinball(actual: np.ndarray, forecast_q: np.ndarray, q: float) -> float:
    a, f = _aligned(actual, forecast_q)
    diff = a - f
    return float(np.where(diff >= 0, q * diff, (1 - q) * (-diff)).mean())


def pinball_multi(
    actual: np.ndarray, quantile_forecasts: dict[float, np.ndarray]
) -> dict[float, float]:
    return {q: pinball(actual, quantile_forecasts[q], q) for q in sorted(quantile_forecasts)}


def coverage(actual: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    a, lo = _aligned(actual, low)
    _, hi = _aligned(actual, high)
    return float(((lo <= a) & (a <= hi)).mean())


def interval_width(low: np.ndarray, high: np.ndarray) -> float:
    lo, hi = _aligned(low, high)
    return float((hi - lo).mean())


def skill_vs_naive(actual: np.ndarray, forecast: np.ndarray, naive_forecast: np.ndarray) -> float:
    """Skill = 1 - WAPE_model / WAPE_naive. Positive means better than naive."""
    w = wape(actual, forecast)
    wn = wape(actual, naive_forecast)
    if wn == 0:
        return 0.0
    return float(1.0 - w / wn)


def scorecard(
    actual: np.ndarray,
    forecast: np.ndarray,
    quantile_forecasts: dict[float, np.ndarray] | None = None,
    naive_forecast: np.ndarray | None = None,
) -> dict[str, float]:
    """Full metric bundle with one schema for every backtest."""
    out: dict[str, float] = {
        "wape": wape(actual, forecast),
        "mae": mae(actual, forecast),
        "bias": bias(actual, forecast),
    }
    if naive_forecast is not None:
        out["mase"] = mase(actual, forecast, naive_forecast)
        out["skill_vs_naive"] = skill_vs_naive(actual, forecast, naive_forecast)
    if quantile_forecasts is not None:
        out["pinball_mean"] = float(
            np.mean(list(pinball_multi(actual, quantile_forecasts).values()))
        )
        for q, fq in quantile_forecasts.items():
            out[f"pinball_q{int(q * 100):02d}"] = pinball(actual, fq, q)
        out["coverage"] = coverage(actual, quantile_forecasts[Q_LOW], quantile_forecasts[Q_HIGH])
        out["interval_width"] = interval_width(
            quantile_forecasts[Q_LOW], quantile_forecasts[Q_HIGH]
        )
    return out
