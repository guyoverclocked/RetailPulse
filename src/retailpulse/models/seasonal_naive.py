"""Classical baselines: last-value and seasonal-naive (lag 7).

The seasonal-naive model is the permanent regression gate — every candidate
must beat it or honestly fail. Both baselines:

- only look at the past, walking back past closed days so an open forecast
  day never inherits a structural zero from a closed reference day
- use the known-future ``Open`` schedule to emit deterministic zeros for
  closed stores (Open is a scheduled input under the prediction-time contract)
- produce P10/P90 via empirical residual quantiles of the same predictor
  evaluated on the training period
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from retailpulse.evaluation.backtester import Model

_Q_LOW, _Q_MID, _Q_HIGH = 0.10, 0.50, 0.90


class SeasonalNaiveModel(Model):
    """Forecast = value 7 days ago (walking back over closed days)."""

    name = "seasonal_naive"

    def fit_predict(
        self,
        train: pd.DataFrame,
        horizon: int,
        valid_dates: list[pd.Timestamp],
        valid_schedule: pd.DataFrame,
    ) -> pd.DataFrame:
        return _naive_predict(train, valid_dates, valid_schedule, lag=7)


class LastValueModel(Model):
    """Forecast = most recent open day's value."""

    name = "last_value"

    def fit_predict(
        self,
        train: pd.DataFrame,
        horizon: int,
        valid_dates: list[pd.Timestamp],
        valid_schedule: pd.DataFrame,
    ) -> pd.DataFrame:
        return _naive_predict(train, valid_dates, valid_schedule, lag=1)


def _reference_value(store_rows: pd.DataFrame, ref_date: pd.Timestamp, lag: int) -> float:
    """Value at the nearest open reference day ≤ ref_date, stepping by lag.

    ``store_rows``: one store's training rows, sorted by Date, with an
    ``Open`` column. Walking back in multiples of ``lag`` preserves weekly
    alignment for lag=7 and daily alignment for lag=1, while skipping closed
    structural zeros.
    """
    past = store_rows[store_rows["Date"] <= ref_date]
    if past.empty:
        return 0.0
    candidate = ref_date
    for _ in range(16):  # look back at most ~16 steps (4 months at lag=7)
        match = past[past["Date"] == candidate]
        if not match.empty:
            row = match.iloc[0]
            if row["Open"] == 1:
                return float(row["Sales"])
            # closed reference: fall back to its value if we run out of steps
            candidate -= pd.Timedelta(days=lag)
            continue
        candidate -= pd.Timedelta(days=lag)
    last = past.iloc[-1]
    return float(last["Sales"])


def _training_residuals(train: pd.DataFrame, lag: int) -> tuple[float, float]:
    """P10/P90 residual quantiles of the predictor on the training period."""
    errors: list[float] = []
    for store, sub in train.groupby("Store"):
        sub = sub.sort_values("Date")
        dates = sub["Date"].to_numpy()
        sales = sub["Sales"].to_numpy(dtype=np.float64)
        opens = sub["Open"].to_numpy(dtype=np.int64)
        for i in range(len(sub)):
            ts = pd.Timestamp(dates[i])
            ref_date = ts - pd.Timedelta(days=lag)
            pred = _reference_value(sub, ref_date, lag)
            if opens[i] == 0:
                pred = 0.0
            errors.append(pred - float(sales[i]))
    if not errors:
        return 0.0, 0.0
    err = np.asarray(errors)
    return float(np.quantile(err, _Q_LOW)), float(np.quantile(err, _Q_HIGH))


def _naive_predict(
    train: pd.DataFrame,
    valid_dates: list[pd.Timestamp],
    valid_schedule: pd.DataFrame,
    *,
    lag: int,
) -> pd.DataFrame:
    train = train.copy()
    train["Date"] = pd.to_datetime(train["Date"])
    train = train.sort_values(["Store", "Date"])

    resid_low, resid_high = _training_residuals(train, lag)
    closed_map = _closed_map(valid_schedule)

    rows: list[dict[str, float | int | pd.Timestamp]] = []
    store_ids = [int(x) for x in np.unique(train["Store"].to_numpy())]
    grouped = {sid: train[train["Store"] == sid] for sid in store_ids}
    for d in valid_dates:
        ref_date = d - pd.Timedelta(days=lag)
        for store, sub in grouped.items():
            is_closed = closed_map.get((store, d.date()), False)
            if is_closed:
                rows.append(
                    {
                        "Store": store,
                        "Date": d,
                        "Sales_q10": 0.0,
                        "Sales_q50": 0.0,
                        "Sales_q90": 0.0,
                    }
                )
                continue
            base = _reference_value(sub, ref_date, lag)
            # Residuals are (pred - actual); a corrected forecast = base - resid.
            rows.append(
                {
                    "Store": store,
                    "Date": d,
                    "Sales_q10": max(base - resid_high, 0.0),
                    "Sales_q50": base - (resid_low + resid_high) / 2.0,
                    "Sales_q90": max(base - resid_low, 0.0),
                }
            )
    return pd.DataFrame(rows)


def _closed_map(valid_schedule: pd.DataFrame) -> dict[tuple[int, object], bool]:
    """(Store, date) -> planned-closed, from the known-future Open column."""
    sched = valid_schedule.copy()
    sched["Date"] = pd.to_datetime(sched["Date"])
    closed = sched[sched["Open"] == 0]
    out: dict[tuple[int, object], bool] = {}
    for store, d in closed[["Store", "Date"]].itertuples(index=False):
        ts = pd.Timestamp(d)
        out[(int(store), ts.date())] = True
    return out
