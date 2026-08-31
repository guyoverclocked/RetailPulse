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
    a, f = _align_many(actual, forecast)
    return a, f


def _align_many(*arrays: np.ndarray) -> list[np.ndarray]:
    """Return all arrays restricted to one shared non-NaN mask."""
    if not arrays:
        raise ValueError("at least one array is required")

    arrs = [np.asarray(array, dtype=np.float64) for array in arrays]
    shapes = {tuple(array.shape) for array in arrs}
    if len(shapes) != 1:
        raise ValueError(f"shape mismatch across inputs: {sorted(shapes)}")

    mask = np.ones(arrs[0].shape, dtype=bool)
    for array in arrs:
        mask &= ~np.isnan(array)
    return [array[mask] for array in arrs]


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
    a, f, nf = _align_many(actual, forecast, naive_forecast)
    if a.size == 0:
        return float("nan")
    denom = np.abs(a - nf).mean()
    if denom == 0:
        return float("nan")
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
    a, lo, hi = _align_many(actual, low, high)
    if a.size == 0:
        return float("nan")
    return float(((lo <= a) & (a <= hi)).mean())


def interval_width(low: np.ndarray, high: np.ndarray) -> float:
    lo, hi = _aligned(low, high)
    if lo.size == 0:
        return float("nan")
    return float((hi - lo).mean())


def skill_vs_naive(actual: np.ndarray, forecast: np.ndarray, naive_forecast: np.ndarray) -> float:
    """Skill = 1 - WAPE_model / WAPE_naive. Positive means better than naive."""
    a, f, nf = _align_many(actual, forecast, naive_forecast)
    if a.size == 0:
        return float("nan")
    actual_denom = np.abs(a).sum()
    if actual_denom == 0:
        return float("nan")
    model_wape = np.abs(a - f).sum() / actual_denom
    naive_wape = np.abs(a - nf).sum() / actual_denom
    if naive_wape == 0:
        return float("nan")
    return float(1.0 - model_wape / naive_wape)


def _validate_group_inputs(
    actual: np.ndarray,
    forecast: np.ndarray,
    group: np.ndarray,
    quantile_forecasts: dict[float, np.ndarray] | None,
    naive_forecast: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[float, np.ndarray], np.ndarray | None]:
    """Validate and normalise arrays used by grouped scorecards."""
    a = np.asarray(actual, dtype=np.float64)
    f = np.asarray(forecast, dtype=np.float64)
    g = np.asarray(group)
    if a.shape != f.shape or a.shape != g.shape:
        raise ValueError(
            f"shape mismatch across actual {a.shape}, forecast {f.shape}, and group {g.shape}"
        )

    qf: dict[float, np.ndarray] = {}
    if quantile_forecasts is not None:
        for q, values in quantile_forecasts.items():
            array = np.asarray(values, dtype=np.float64)
            if array.shape != a.shape:
                raise ValueError(f"shape mismatch for quantile {q}: {array.shape} vs {a.shape}")
            qf[q] = array

    nf: np.ndarray | None = None
    if naive_forecast is not None:
        nf = np.asarray(naive_forecast, dtype=np.float64)
        if nf.shape != a.shape:
            raise ValueError(f"shape mismatch for naive forecast: {nf.shape} vs {a.shape}")
    return a, f, g, qf, nf


def by_horizon(
    actual: np.ndarray,
    forecast: np.ndarray,
    horizon_step: np.ndarray,
    *,
    quantile_forecasts: dict[float, np.ndarray] | None = None,
    naive_forecast: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """Compute scorecards for each horizon step.

    The returned rows are JSON-friendly and use ``h`` and ``n`` alongside the
    regular scorecard metrics, making them suitable for later report wiring.
    """
    a, f, h, qf, nf = _validate_group_inputs(
        actual, forecast, horizon_step, quantile_forecasts, naive_forecast
    )
    values = sorted(value for value in np.unique(h) if not _is_nan_group(value))
    rows: list[dict[str, float]] = []
    for value in values:
        mask = h == value
        row: dict[str, float] = {"h": float(value), "n": float(mask.sum())}
        row.update(
            scorecard(
                a[mask],
                f[mask],
                quantile_forecasts={q: values_[mask] for q, values_ in qf.items()} if qf else None,
                naive_forecast=nf[mask] if nf is not None else None,
            )
        )
        rows.append(row)
    return rows


def by_segment(
    actual: np.ndarray,
    forecast: np.ndarray,
    segment: np.ndarray,
    *,
    quantile_forecasts: dict[float, np.ndarray] | None = None,
    naive_forecast: np.ndarray | None = None,
) -> list[dict[str, float | str]]:
    """Compute scorecards for each value in one categorical segment."""
    a, f, segment_values, qf, nf = _validate_group_inputs(
        actual, forecast, segment, quantile_forecasts, naive_forecast
    )
    values = sorted(value for value in np.unique(segment_values) if not _is_nan_group(value))
    rows: list[dict[str, float | str]] = []
    for value in values:
        mask = segment_values == value
        row: dict[str, float | str] = {"value": str(value), "n": float(mask.sum())}
        row.update(
            scorecard(
                a[mask],
                f[mask],
                quantile_forecasts={q: values_[mask] for q, values_ in qf.items()} if qf else None,
                naive_forecast=nf[mask] if nf is not None else None,
            )
        )
        rows.append(row)
    return rows


def _is_nan_group(value: object) -> bool:
    """Return whether a group label is a floating-point NaN."""
    return isinstance(value, (float, np.floating)) and bool(np.isnan(value))


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
