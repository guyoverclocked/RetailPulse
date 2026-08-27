"""Backtest model protocol and runner.

Every candidate model implements :class:`Model` and is scored by the
:class:`Backtester` on identical folds. Models receive:

- ``train``: a DataFrame of store-day rows up to the fold origin
- ``horizon``: the forecast horizon (e.g. 30 days)
- ``valid_dates``: the dates to forecast
- ``valid_schedule``: known-future columns per (Store, Date) — ``Open``,
  ``Promo``, ``StateHoliday``, ``SchoolHoliday`` — the only future inputs the
  prediction-time contract allows

and must return a DataFrame with columns ``Store, Date, Sales_q10, Sales_q50,
Sales_q90`` (point forecast = ``Sales_q50``). Closed stores must be forecast as
deterministic zeros.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from retailpulse.evaluation import metrics
from retailpulse.evaluation.splits import Fold, RollingOriginSplitter, assert_no_leakage

SCHEDULE_COLUMNS = ["Open", "Promo", "StateHoliday", "SchoolHoliday"]


class Model(Protocol):
    name: str

    def fit_predict(
        self,
        train: pd.DataFrame,
        horizon: int,
        valid_dates: list[pd.Timestamp],
        valid_schedule: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fit on train (all rows ≤ origin) and forecast the valid dates."""
        ...


@dataclass
class FoldResult:
    """One fold's outcome."""

    fold_id: int
    model: str
    scorecard: dict[str, float]
    runtime_seconds: float
    n_train_rows: int
    n_valid_rows: int
    failed: bool = False
    error: str | None = None


@dataclass
class BacktestResult:
    """Aggregate outcome across folds."""

    model: str
    folds: list[FoldResult] = field(default_factory=list)
    aggregate: dict[str, float] = field(default_factory=dict)


class Backtester:
    """Runs any Model through identical rolling-origin folds."""

    def __init__(
        self, splitter: RollingOriginSplitter, *, quantiles: tuple[float, ...] = (0.10, 0.50, 0.90)
    ) -> None:
        self.splitter = splitter
        self.quantiles = quantiles

    def evaluate(
        self,
        model: Model,
        data: pd.DataFrame,
        *,
        naive: Model | None = None,
    ) -> BacktestResult:
        """Run the model across all folds and return scored results.

        ``naive`` (the seasonal-naive baseline) may be passed so MASE and skill
        are computed on the same folds.
        """
        result = BacktestResult(model=model.name)
        for fold in self.splitter.folds():
            result.folds.append(self._run_fold(model, data, fold, naive=naive))
        result.aggregate = _aggregate(result.folds)
        return result

    def _run_fold(
        self,
        model: Model,
        data: pd.DataFrame,
        fold: Fold,
        *,
        naive: Model | None,
    ) -> FoldResult:
        train = data[self.splitter.train_mask(data, fold)]
        assert_no_leakage(train, fold)
        valid = data[self.splitter.valid_mask(data, fold)]
        valid_dates = _valid_date_list(fold)
        # Known-future schedule: ONLY the columns the availability contract
        # allows (Open, Promo, StateHoliday, SchoolHoliday). Target and
        # observed-late columns are excluded by construction.
        valid_schedule = valid[["Store", "Date", *SCHEDULE_COLUMNS]].copy()
        try:
            start = time.perf_counter()
            pred = model.fit_predict(train, self.splitter.horizon_days, valid_dates, valid_schedule)
            elapsed = time.perf_counter() - start

            naive_pred = None
            if naive is not None:
                naive_pred = naive.fit_predict(
                    train, self.splitter.horizon_days, valid_dates, valid_schedule
                )

            merged = _merge_on_store_date(valid, pred, naive_pred)

            if merged.empty:
                return FoldResult(
                    fold_id=fold.fold_id,
                    model=model.name,
                    scorecard={},
                    runtime_seconds=elapsed,
                    n_train_rows=len(train),
                    n_valid_rows=len(valid),
                    failed=True,
                    error="no overlapping store-date rows between predictions and validation",
                )

            quantile_preds = {
                q: merged[f"Sales_q{int(q * 100):02d}"].to_numpy() for q in self.quantiles
            }
            naive_q50 = merged["naive_q50"].to_numpy() if naive_pred is not None else None
            sc = metrics.scorecard(
                merged["Sales"].to_numpy(),
                merged["Sales_q50"].to_numpy(),
                quantile_forecasts=quantile_preds,
                naive_forecast=naive_q50,
            )
            return FoldResult(
                fold_id=fold.fold_id,
                model=model.name,
                scorecard=sc,
                runtime_seconds=elapsed,
                n_train_rows=len(train),
                n_valid_rows=len(valid),
            )
        except Exception as exc:
            return FoldResult(
                fold_id=fold.fold_id,
                model=model.name,
                scorecard={},
                runtime_seconds=0.0,
                n_train_rows=len(train),
                n_valid_rows=len(valid),
                failed=True,
                error=str(exc),
            )


def _valid_date_list(fold: Fold) -> list[pd.Timestamp]:
    return list(pd.date_range(fold.valid_dates[0], fold.valid_dates[1], freq="D"))


def _merge_on_store_date(
    valid: pd.DataFrame, pred: pd.DataFrame, naive_pred: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Inner-join predictions onto validation rows by (Store, Date)."""
    key = ["Store", "Date"]
    pred_cols = [c for c in pred.columns if c.startswith("Sales_q")] + key
    left = valid.copy()
    left["Date"] = pd.to_datetime(left["Date"])
    pred = pred.copy()
    pred["Date"] = pd.to_datetime(pred["Date"])
    out = left.merge(pred[pred_cols], on=key, how="inner")
    if naive_pred is not None:
        n = naive_pred.copy()
        n["Date"] = pd.to_datetime(n["Date"])
        out = out.merge(
            n[[*key, "Sales_q50"]].rename(columns={"Sales_q50": "naive_q50"}),
            on=key,
            how="inner",
        )
    return out


def _aggregate(folds: list[FoldResult]) -> dict[str, float]:
    agg: dict[str, float] = {}
    keys = (
        "wape",
        "mae",
        "bias",
        "mase",
        "skill_vs_naive",
        "pinball_mean",
        "coverage",
        "interval_width",
    )
    for k in keys:
        vals = [f.scorecard[k] for f in folds if k in f.scorecard and not f.failed]
        if vals:
            agg[k] = float(np.mean(vals))
    return agg
