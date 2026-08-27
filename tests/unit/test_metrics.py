"""Metrics unit tests with hand-computed values."""

from __future__ import annotations

import numpy as np
import pytest

from retailpulse.evaluation import metrics


def test_wape_hand_computed() -> None:
    actual = np.array([100.0, 200.0, 0.0])
    forecast = np.array([90.0, 220.0, 0.0])
    # |100-90| + |200-220| + |0-0| = 10 + 20 + 0 = 30; sum|actual| = 300
    assert metrics.wape(actual, forecast) == pytest.approx(0.1)


def test_mae_and_bias() -> None:
    actual = np.array([10.0, 20.0])
    forecast = np.array([12.0, 16.0])
    assert metrics.mae(actual, forecast) == 3.0
    assert metrics.bias(actual, forecast) == -1.0  # under-forecasting


def test_pinball() -> None:
    actual = np.array([10.0, 10.0])
    fq = np.array([8.0, 12.0])
    # q=0.5: |10-8|*0.5 + |10-12|*0.5 = 1 + 1 = 2; mean = 1
    assert metrics.pinball(actual, fq, 0.5) == 1.0
    # q=0.1: first over (10>=8) => 0.1*2=0.2; second under => 0.9*2=1.8; mean=1.0
    assert metrics.pinball(actual, fq, 0.1) == 1.0
    # q=0.9: first over => 0.9*2=1.8; second under => 0.1*2=0.2; mean=1.0
    assert metrics.pinball(actual, fq, 0.9) == 1.0


def test_pinball_asymmetry() -> None:
    actual = np.array([10.0])
    low_f = np.array([9.0])  # under-forecast at q=0.9 is heavily penalized
    high_f = np.array([11.0])
    assert metrics.pinball(actual, low_f, 0.9) > metrics.pinball(actual, high_f, 0.9)


def test_coverage_and_width() -> None:
    actual = np.array([5.0, 15.0, 10.0])
    low = np.array([0.0, 10.0, 0.0])
    high = np.array([10.0, 20.0, 20.0])
    # inside: 5 in [0,10] yes; 15 in [10,20] yes; 10 in [0,20] yes => 3/3
    assert metrics.coverage(actual, low, high) == 1.0
    assert metrics.interval_width(low, high) == 40.0 / 3.0


def test_mase() -> None:
    actual = np.array([10.0, 20.0, 30.0])
    model = np.array([9.0, 19.0, 29.0])
    naive = np.array([11.0, 21.0, 31.0])
    # MAE_model=1, MAE_naive=1 => MASE=1
    assert metrics.mase(actual, model, naive) == 1.0


def test_skill() -> None:
    actual = np.array([10.0, 20.0, 30.0])
    model = np.array([9.0, 19.0, 29.0])
    naive = np.array([11.0, 21.0, 31.0])
    assert metrics.skill_vs_naive(actual, model, naive) == 0.0


def test_zero_sales_wape_no_blowup() -> None:
    actual = np.array([0.0, 0.0])
    forecast = np.array([0.0, 5.0])
    # denom 0, forecast nonzero => inf (defined, not NaN)
    assert metrics.wape(actual, forecast) == float("inf")


def test_nan_pairs_skipped() -> None:
    actual = np.array([10.0, np.nan, 20.0])
    forecast = np.array([12.0, 99.0, 16.0])
    assert metrics.mae(actual, forecast) == 3.0
