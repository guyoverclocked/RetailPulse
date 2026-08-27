"""LightGBM quantile candidate integration test on the CI fixture."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from retailpulse.evaluation.backtester import Backtester
from retailpulse.evaluation.splits import RollingOriginSplitter
from retailpulse.models.lightgbm_quantile import LightGBMQuantileModel
from retailpulse.models.seasonal_naive import SeasonalNaiveModel

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "sample"


@pytest.fixture(scope="module")
def fixture_data() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / "train.csv", parse_dates=["Date"])


def test_lightgbm_candidate_runs_and_scores(fixture_data: pd.DataFrame) -> None:
    splitter = RollingOriginSplitter(horizon_days=30, n_folds=2)
    splitter.fit(fixture_data["Date"])
    result = Backtester(splitter).evaluate(
        LightGBMQuantileModel(n_estimators=50, num_leaves=15, early_stopping_rounds=0),
        fixture_data,
    )
    assert not any(f.failed for f in result.folds), [f.error for f in result.folds]
    assert "wape" in result.aggregate
    assert "coverage" in result.aggregate


def test_lightgbm_beats_naive_on_fixture(fixture_data: pd.DataFrame) -> None:
    """The candidate should beat seasonal-naive on the synthetic fixture,
    where lag features carry real signal."""
    splitter = RollingOriginSplitter(horizon_days=30, n_folds=2)
    splitter.fit(fixture_data["Date"])
    backtester = Backtester(splitter)
    naive = backtester.evaluate(SeasonalNaiveModel(), fixture_data)
    candidate = backtester.evaluate(
        LightGBMQuantileModel(n_estimators=100, num_leaves=31, early_stopping_rounds=0),
        fixture_data,
    )
    assert candidate.aggregate["wape"] < naive.aggregate["wape"]


def test_quantiles_are_ordered(fixture_data: pd.DataFrame) -> None:
    splitter = RollingOriginSplitter(horizon_days=30, n_folds=1)
    splitter.fit(fixture_data["Date"])
    fold = splitter.folds()[0]
    train = fixture_data[splitter.train_mask(fixture_data, fold)]
    sched = fixture_data[splitter.valid_mask(fixture_data, fold)][
        ["Store", "Date", "Open", "Promo", "StateHoliday", "SchoolHoliday"]
    ]
    valid_dates = list(pd.date_range(fold.valid_dates[0], fold.valid_dates[1], freq="D"))
    model = LightGBMQuantileModel(n_estimators=30, num_leaves=15, early_stopping_rounds=0)
    pred = model.fit_predict(train, 30, valid_dates, sched)
    assert (pred["Sales_q10"] <= pred["Sales_q50"]).all()
    assert (pred["Sales_q50"] <= pred["Sales_q90"]).all()
    # Closed stores are deterministic zeros.
    closed = sched[sched["Open"] == 0][["Store", "Date"]]
    merged = pred.merge(closed.assign(closed=1), on=["Store", "Date"], how="left")
    if (merged["closed"] == 1).any():
        closed_preds = merged[merged["closed"] == 1]
        assert (closed_preds["Sales_q50"] == 0).all()
