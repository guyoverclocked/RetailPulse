"""Quantile calibration: crossing repair and empirical checks.

P10 ≤ P50 ≤ P90 must hold for every store-day. Crossings happen when three
independent quantile models disagree; the repair is a monotone sort of the
three values per row. Coverage and width are then reported against the
repaired intervals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def repair_crossings(
    q10: np.ndarray, q50: np.ndarray, q90: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enforce q10 ≤ q50 ≤ q90 by sorting each row's three values."""
    stacked = np.stack([q10, q50, q90], axis=1)
    sorted_ = np.sort(stacked, axis=1)
    return sorted_[:, 0], sorted_[:, 1], sorted_[:, 2]


def repair_crossings_df(df: pd.DataFrame) -> pd.DataFrame:
    """Repair quantile crossings on a forecast frame in place (returns copy)."""
    out = df.copy()
    low, mid, high = repair_crossings(
        out["Sales_q10"].to_numpy(dtype=np.float64),
        out["Sales_q50"].to_numpy(dtype=np.float64),
        out["Sales_q90"].to_numpy(dtype=np.float64),
    )
    out["Sales_q10"] = low
    out["Sales_q50"] = mid
    out["Sales_q90"] = high
    return out


def crossing_rate(df: pd.DataFrame) -> float:
    """Fraction of rows where quantiles were out of order before repair."""
    q10 = df["Sales_q10"].to_numpy(dtype=np.float64)
    q50 = df["Sales_q50"].to_numpy(dtype=np.float64)
    q90 = df["Sales_q90"].to_numpy(dtype=np.float64)
    violations = (q10 > q50) | (q50 > q90)
    return float(violations.mean())


def calibration_summary(actual: np.ndarray, low: np.ndarray, high: np.ndarray) -> dict[str, float]:
    """Empirical coverage and width of a P10-P90 band."""
    a = np.asarray(actual, dtype=np.float64)
    lo = np.asarray(low, dtype=np.float64)
    hi = np.asarray(high, dtype=np.float64)
    mask = ~(np.isnan(a) | np.isnan(lo) | np.isnan(hi))
    a, lo, hi = a[mask], lo[mask], hi[mask]
    return {
        "coverage": float(((lo <= a) & (a <= hi)).mean()),
        "interval_width": float((hi - lo).mean()),
        "n": len(a),
    }


def conformal_corrections(
    actual: np.ndarray,
    q10_pred: np.ndarray,
    q90_pred: np.ndarray,
    *,
    miscoverage: float = 0.20,
) -> tuple[float, float]:
    """Conformal quantile corrections for a P10-P90 band.

    Conformity scores ``s_lo = q10 - y`` and ``s_hi = y - q90`` measured on a
    calibration slice; returning corrections so that the band
    ``[q10 - corr_lo, q90 + corr_hi]`` achieves roughly
    ``1 - miscoverage`` coverage (symmetric tails). Calibration data must
    never include the forecast horizon.
    """
    a = np.asarray(actual, dtype=np.float64)
    lo = np.asarray(q10_pred, dtype=np.float64)
    hi = np.asarray(q90_pred, dtype=np.float64)
    mask = ~(np.isnan(a) | np.isnan(lo) | np.isnan(hi))
    if mask.sum() == 0:
        return 0.0, 0.0
    s_lo = lo[mask] - a[mask]
    s_hi = a[mask] - hi[mask]
    tail = 1.0 - miscoverage / 2.0  # 0.90 for a 20% band
    corr_lo = float(np.quantile(s_lo, tail))
    corr_hi = float(np.quantile(s_hi, tail))
    return corr_lo, corr_hi
