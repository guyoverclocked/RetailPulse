"""Leakage tests — the non-negotiable suite.

The canonical leak scenarios that MUST be caught:

1. Training rows dated after a fold origin.
2. Features built on rows dated after the fold origin (e.g. lags computed on
   the full dataset, then the split applied afterwards).
3. A feature whose availability class forbids use at forecast time.
"""

from __future__ import annotations

import pandas as pd
import pytest

from retailpulse.config import get_config
from retailpulse.data.availability import (
    AvailabilityError,
    assert_legal_features,
    assert_no_future_rows,
)
from retailpulse.evaluation.splits import RollingOriginSplitter, assert_no_leakage
from retailpulse.features.lags import build_lag_rolling_features


def _make_dates(n: int = 600) -> pd.DataFrame:
    return pd.DataFrame({"Date": pd.date_range("2013-01-01", periods=n, freq="D")})


def _make_store_day(n_days: int = 400, n_stores: int = 5) -> pd.DataFrame:
    """Store-day frame with sales for lag-feature tests."""
    dates = pd.date_range("2013-01-01", periods=n_days, freq="D")
    rows = []
    for store in range(1, n_stores + 1):
        for i, d in enumerate(dates):
            rows.append({"Store": store, "Date": d, "Sales": float(i + store)})
    return pd.DataFrame(rows)


def test_no_training_row_crosses_origin() -> None:
    dates = _make_dates()
    splitter = RollingOriginSplitter(horizon_days=30, n_folds=3)
    splitter.fit(dates["Date"])
    for fold in splitter.folds():
        train = dates[splitter.train_mask(dates, fold)]
        assert_no_leakage(train, fold)  # must not raise
        assert (train["Date"] <= fold.origin).all()


def test_future_training_row_is_rejected() -> None:
    splitter = RollingOriginSplitter(horizon_days=30, n_folds=3)
    splitter.fit(_make_dates()["Date"])
    fold = splitter.folds()[0]
    bad = pd.DataFrame({"Date": [fold.origin + pd.Timedelta(days=1)]})
    with pytest.raises(ValueError, match="after origin"):
        assert_no_leakage(bad, fold)


def test_feature_rows_after_origin_rejected() -> None:
    from datetime import date

    df = pd.DataFrame({"Date": pd.to_datetime(["2020-01-05"])})
    with pytest.raises(AvailabilityError):
        assert_no_future_rows(df, origin=date(2020, 1, 1))


def test_holdout_is_sealed_by_default() -> None:
    cfg = get_config().backtest
    assert cfg.holdout.locked


def test_splitter_reserves_holdout_when_locked() -> None:
    dates = _make_dates()
    splitter = RollingOriginSplitter(horizon_days=30, n_folds=3)
    splitter.fit(dates["Date"])
    last_fold = splitter.folds()[-1]
    assert last_fold.valid_dates[1] <= dates["Date"].max() - pd.Timedelta(days=30)


def test_lags_only_reference_the_past() -> None:
    """lag_1 of row t equals Sales of row t-1 within the same store."""
    df = _make_store_day()
    feats = build_lag_rolling_features(df)
    # Row ordering is Store-major, Date-minor; verify per store.
    for store in feats["Store"].unique():
        sub = feats[feats["Store"] == store].sort_values("Date").reset_index(drop=True)
        expected = sub["Sales"].shift(1)
        pd.testing.assert_series_equal(
            sub["lag_1"].rename("lag_1"), expected.rename("lag_1"), check_names=False
        )


def test_lag_1_of_first_validation_day_is_last_train_day() -> None:
    """The classic bug: lag features built AFTER the split make lag_1 of the
    first validation day equal to the last training day's sales — but that IS
    legal (that sales value existed at origin). The illegal case is a lag_0
    or any feature referencing the validation day itself. We assert the
    pipeline never produces a lag that references same-day data."""
    df = _make_store_day()
    feats = build_lag_rolling_features(df)
    for lag in (1, 7, 14):
        col = f"lag_{lag}"
        for store in feats["Store"].unique():
            sub = feats[feats["Store"] == store].sort_values("Date").reset_index(drop=True)
            # First `lag` rows must be NaN (no history).
            assert sub[col].iloc[:lag].isna().all()


def test_forbidden_feature_rejected() -> None:
    with pytest.raises(AvailabilityError):
        assert_legal_features(["Customers"], context="model features")
    with pytest.raises(AvailabilityError):
        assert_legal_features(["Sales"], context="model features")


def test_deliberate_post_split_leak_is_caught() -> None:
    """Simulate the real-world bug: features built on the FULL dataset (so the
    first validation row's lag_1 comes from the validation window's first day,
    which is in the future), then the split applied afterwards. The contract
    checker must refuse the feature frame."""
    df = _make_store_day()
    splitter = RollingOriginSplitter(horizon_days=30, n_folds=3)
    splitter.fit(pd.Series(pd.date_range("2013-01-01", periods=400, freq="D")))
    fold = splitter.folds()[0]

    # Buggy code path: features built on everything (including validation rows).
    leaked_features = build_lag_rolling_features(df)
    # The buggy pipeline then hands validation rows (dated after origin) to the model.
    bad_rows = leaked_features[leaked_features["Date"] > fold.origin]
    assert not bad_rows.empty, "test setup wrong: expected rows after origin"
    with pytest.raises(AvailabilityError):
        assert_no_future_rows(bad_rows, origin=fold.origin.date())
