"""Backtester integration: seasonal-naive and last-value on the CI fixture."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from retailpulse.evaluation.backtester import Backtester
from retailpulse.evaluation.splits import RollingOriginSplitter
from retailpulse.models.seasonal_naive import LastValueModel, SeasonalNaiveModel

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "sample"


def _load_fixture() -> pd.DataFrame:
    df = pd.read_csv(FIXTURE_DIR / "train.csv", parse_dates=["Date"])
    return df


def test_seasonal_naive_backtest_runs_and_scores() -> None:
    data = _load_fixture()
    splitter = RollingOriginSplitter(horizon_days=30, n_folds=2)
    splitter.fit(data["Date"])
    backtester = Backtester(splitter)
    result = backtester.evaluate(SeasonalNaiveModel(), data)
    assert not any(f.failed for f in result.folds), [f.error for f in result.folds]
    assert "wape" in result.aggregate
    assert result.aggregate["wape"] >= 0
    assert "coverage" in result.aggregate


def test_last_value_backtest_runs() -> None:
    data = _load_fixture()
    splitter = RollingOriginSplitter(horizon_days=30, n_folds=2)
    splitter.fit(data["Date"])
    result = Backtester(splitter).evaluate(LastValueModel(), data)
    assert not any(f.failed for f in result.folds)


def test_naive_comparison_includes_mase_and_skill() -> None:
    data = _load_fixture()
    splitter = RollingOriginSplitter(horizon_days=30, n_folds=2)
    splitter.fit(data["Date"])
    result = Backtester(splitter).evaluate(LastValueModel(), data, naive=SeasonalNaiveModel())
    assert "mase" in result.aggregate
    assert "skill_vs_naive" in result.aggregate


def test_identical_folds_across_models() -> None:
    """Both models must see the same fold boundaries."""
    data = _load_fixture()
    s1 = RollingOriginSplitter(horizon_days=30, n_folds=2)
    s2 = RollingOriginSplitter(horizon_days=30, n_folds=2)
    s1.fit(data["Date"])
    s2.fit(data["Date"])
    assert [f.origin for f in s1.folds()] == [f.origin for f in s2.folds()]
