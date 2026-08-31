"""Tests for chronological, gap-aware rolling-origin folds."""

from __future__ import annotations

import pandas as pd
import pytest

from retailpulse.evaluation.splits import RollingOriginSplitter


def test_fold_ids_are_chronological() -> None:
    splitter = RollingOriginSplitter(horizon_days=30, n_folds=3, min_train_days=30)
    splitter.fit(pd.Series(pd.date_range("2013-01-01", periods=600, freq="D")))

    folds = splitter.folds()

    assert [fold.fold_id for fold in folds] == [0, 1, 2]
    assert [fold.origin for fold in folds] == sorted(fold.origin for fold in folds)
    assert splitter.n_effective_folds == 3


def test_short_history_keeps_the_folds_that_fit() -> None:
    splitter = RollingOriginSplitter(horizon_days=30, n_folds=4, min_train_days=60)
    splitter.fit(pd.Series(pd.date_range("2014-01-01", periods=150, freq="D")))

    assert 0 < len(splitter.folds()) < 4
    assert splitter.n_effective_folds == len(splitter.folds())


def test_gap_days_is_honoured() -> None:
    splitter = RollingOriginSplitter(horizon_days=10, n_folds=2, min_train_days=30, gap_days=3)
    dates = pd.Series(pd.date_range("2020-01-01", periods=120, freq="D"))
    splitter.fit(dates)

    for fold in splitter.folds():
        assert fold.valid_dates[0] == fold.origin + pd.Timedelta(days=4)
        assert len(pd.date_range(fold.valid_dates[0], fold.valid_dates[1], freq="D")) == 10


def test_min_train_days_is_honoured() -> None:
    splitter = RollingOriginSplitter(horizon_days=10, n_folds=2, min_train_days=50)
    splitter.fit(pd.Series(pd.date_range("2020-01-01", periods=150, freq="D")))

    for fold in splitter.folds():
        n_train_days = (fold.origin - fold.train_dates[0]).days + 1
        assert n_train_days >= 50


def test_no_fit_message_reports_the_constraint() -> None:
    splitter = RollingOriginSplitter(horizon_days=30, n_folds=2, min_train_days=365)

    with pytest.raises(ValueError, match=r"no fold fits.*365.*2x\(30\+0\)"):
        splitter.fit(pd.Series(pd.date_range("2020-01-01", periods=100, freq="D")))
