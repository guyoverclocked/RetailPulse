"""Calibration tests: crossing repair and coverage."""

from __future__ import annotations

import numpy as np

from retailpulse.calibration.quantile import (
    calibration_summary,
    crossing_rate,
    repair_crossings,
    repair_crossings_df,
)


def test_repair_crossings_sorts_rows() -> None:
    q10 = np.array([5.0, 9.0])
    q50 = np.array([3.0, 8.0])
    q90 = np.array([4.0, 10.0])
    lo, mid, hi = repair_crossings(q10, q50, q90)
    assert (lo <= mid).all()
    assert (mid <= hi).all()
    # First row sorted: [3,4,5]; second: [8,9,10]
    np.testing.assert_array_equal(lo, [3.0, 8.0])
    np.testing.assert_array_equal(mid, [4.0, 9.0])
    np.testing.assert_array_equal(hi, [5.0, 10.0])


def test_repair_never_widens_already_sorted() -> None:
    q10 = np.array([1.0, 2.0])
    q50 = np.array([3.0, 4.0])
    q90 = np.array([5.0, 6.0])
    lo, mid, hi = repair_crossings(q10, q50, q90)
    np.testing.assert_array_equal(lo, q10)
    np.testing.assert_array_equal(mid, q50)
    np.testing.assert_array_equal(hi, q90)


def test_calibration_summary() -> None:
    actual = np.array([5.0, 15.0, 50.0])
    low = np.array([0.0, 0.0, 40.0])
    high = np.array([10.0, 20.0, 60.0])
    s = calibration_summary(actual, low, high)
    assert s["n"] == 3
    assert s["coverage"] == 1.0
    assert s["interval_width"] == 50.0 / 3.0


def test_repair_crossings_df() -> None:
    import pandas as pd

    df = pd.DataFrame({"Sales_q10": [5.0], "Sales_q50": [3.0], "Sales_q90": [4.0]})
    assert crossing_rate(df) == 1.0
    fixed = repair_crossings_df(df)
    assert crossing_rate(fixed) == 0.0
    assert fixed["Sales_q10"].iloc[0] == 3.0
