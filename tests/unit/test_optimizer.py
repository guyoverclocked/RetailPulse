"""Optimizer constraint-satisfaction and simulator unit tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from retailpulse.optimization.optimizer import optimize_staffing
from retailpulse.optimization.simulator import (
    demand_to_required,
    evaluate_allocation,
    required_hours,
)


def _small_frame(n_stores: int = 4, n_days: int = 7) -> pd.DataFrame:
    rows = []
    for store in range(1, n_stores + 1):
        for d in range(n_days):
            open_flag = 0 if (store == 3 and d == 3) else 1
            rows.append(
                {
                    "Store": store,
                    "Date": pd.Timestamp("2015-06-01") + pd.Timedelta(days=d),
                    "Open": open_flag,
                    "Sales": float(100 * store) if open_flag else 0.0,
                }
            )
    df = pd.DataFrame(rows)
    return demand_to_required(df)


def test_required_hours_respects_closure() -> None:
    req = required_hours(
        np.array([100.0, 50.0]),
        np.array([1, 0]),
        productivity=100.0,
        base_hours=4.0,
    )
    assert req[0] == pytest.approx(5.0)  # 4 + 100/100
    assert req[1] == 0.0


def test_evaluate_allocation_hand_computed() -> None:
    r = evaluate_allocation(
        np.array([10.0, 10.0]),
        np.array([8.0, 12.0]),
        wage=15.0,
        under_penalty=45.0,
        over_penalty=22.5,
    )
    # labor = 15*(8+12)=300; under=2*45=90; over=2*22.5=45
    assert r.total_cost == pytest.approx(435.0)
    assert r.understaff_hours == pytest.approx(2.0)
    assert r.overstaff_hours == pytest.approx(2.0)


def test_optimizer_never_violates_closed_store_rule() -> None:
    frame = _small_frame()
    result = optimize_staffing(frame)
    assert not result.infeasible
    alloc = result.allocation
    closed = frame[frame["Open"] == 0][["Store", "Date"]]
    merged = alloc.merge(closed.assign(closed=1), on=["Store", "Date"], how="left")
    if (merged["closed"] == 1).any():
        closed_rows = merged[merged["closed"] == 1]
        assert (closed_rows["scheduled_hours"] == 0).all()


def test_optimizer_respects_min_hours_for_open_stores() -> None:
    frame = _small_frame()
    result = optimize_staffing(frame, min_hours=4.0, budget_hours=1e9)
    alloc = result.allocation
    open_rows = alloc[alloc["required_hours"] > 0]
    assert (open_rows["scheduled_hours"] >= 4.0).all()


def test_optimizer_respects_budget() -> None:
    frame = _small_frame()
    result = optimize_staffing(frame, budget_hours=100.0)
    alloc = result.allocation
    per_day = alloc.groupby("Date")["scheduled_hours"].sum()
    assert (per_day <= 100.0 + 1e-6).all()


def test_optimizer_infeasible_detected() -> None:
    frame = _small_frame()
    # Impossible: budget below 4 stores x 4 min hours = 16.
    result = optimize_staffing(frame, budget_hours=5.0, min_hours=4.0)
    assert result.infeasible
    assert result.diagnostics


def test_allocation_covers_every_store_day() -> None:
    frame = _small_frame()
    result = optimize_staffing(frame)
    assert len(result.allocation) == len(frame)
