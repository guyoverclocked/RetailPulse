"""Staffing policies: fixed rule, P50 plan, P90 plan, budget sweep.

Each policy converts a forecast frame into an allocation; the simulator
evaluates every policy against the SAME actual demand so comparisons are
honest. Policies:

- ``fixed_rule``: constant hours per open store (the simple baseline).
- ``p50_plan``:  optimizer fed with P50 forecasts.
- ``p90_plan``:  optimizer fed with P90 forecasts (risk-averse).
- ``budget_sweep``: P50 plans under a range of labor budgets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from retailpulse.optimization.optimizer import optimize_staffing
from retailpulse.optimization.simulator import demand_to_required, evaluate_allocation


@dataclass(frozen=True)
class PolicyResult:
    """One policy's outcome against actual demand."""

    policy: str
    total_cost: float
    labor_cost: float
    understaff_cost: float
    overstaff_cost: float
    budget_hours: float | None = None


def fixed_rule(
    frame: pd.DataFrame,
    *,
    hours_per_open_store: float = 8.0,
    store_col: str = "Store",
    date_col: str = "Date",
    open_col: str = "Open",
) -> pd.DataFrame:
    """Constant-hours-per-open-store allocation."""
    out = frame[[store_col, date_col]].copy()
    out["scheduled_hours"] = np.where(frame[open_col].to_numpy() == 1, hours_per_open_store, 0.0)
    return out


def p50_plan(forecast: pd.DataFrame, *, budget_hours: float | None = None) -> pd.DataFrame:
    """Optimizer fed with P50 forecasts as workload."""
    frame = demand_to_required(forecast.rename(columns={"Sales_q50": "Sales"}))
    return optimize_staffing(frame, budget_hours=budget_hours).allocation


def p90_plan(forecast: pd.DataFrame, *, budget_hours: float | None = None) -> pd.DataFrame:
    """Optimizer fed with P90 forecasts (risk-averse workload)."""
    frame = demand_to_required(forecast.rename(columns={"Sales_q90": "Sales"}))
    return optimize_staffing(frame, budget_hours=budget_hours).allocation


def compare_policies(
    forecast: pd.DataFrame,
    actual_demand: pd.DataFrame,
    *,
    budget_hours: float | None = None,
    fixed_hours: float = 8.0,
) -> list[PolicyResult]:
    """Evaluate all policies against the same actual demand.

    Args:
        forecast: store-day frame with Sales_q50/Sales_q90/Open columns.
        actual_demand: store-day frame with Sales (actual) and Open columns.
    """
    from retailpulse.config import get_config

    cfg = get_config().staffing.assumptions
    wage = float(cfg["wage_per_hour"])
    under_pen = float(cfg["understaff_penalty_per_hour"])
    over_pen = float(cfg["overstaff_penalty_per_hour"])

    actual_req = demand_to_required(actual_demand)["required_hours"].to_numpy(dtype=np.float64)

    def cost_of(allocation: pd.DataFrame) -> tuple[float, float, float, float]:
        merged = actual_demand[["Store", "Date"]].merge(
            allocation, on=["Store", "Date"], how="left"
        )
        sched = merged["scheduled_hours"].fillna(0).to_numpy(dtype=np.float64)
        r = evaluate_allocation(
            actual_req, sched, wage=wage, under_penalty=under_pen, over_penalty=over_pen
        )
        return r.total_cost, r.labor_cost, r.understaff_cost, r.overstaff_cost

    results: list[PolicyResult] = []

    fixed = fixed_rule(forecast, hours_per_open_store=fixed_hours)
    c = cost_of(fixed)
    results.append(PolicyResult("fixed_rule", c[0], c[1], c[2], c[3], budget_hours=None))

    for name, plan_fn in (("p50_plan", p50_plan), ("p90_plan", p90_plan)):
        try:
            alloc = plan_fn(forecast, budget_hours=budget_hours)
            c = cost_of(alloc)
            results.append(PolicyResult(name, c[0], c[1], c[2], c[3], budget_hours=budget_hours))
        except Exception:
            results.append(
                PolicyResult(name, float("inf"), 0.0, 0.0, 0.0, budget_hours=budget_hours)
            )

    return results


def budget_sweep(
    forecast: pd.DataFrame,
    actual_demand: pd.DataFrame,
    budgets: list[float],
) -> list[PolicyResult]:
    """P50 plans under a range of chain-wide budgets."""
    out: list[PolicyResult] = []
    for b in budgets:
        r = compare_policies(forecast, actual_demand, budget_hours=b)
        p50 = next(x for x in r if x.policy == "p50_plan")
        out.append(
            PolicyResult(
                policy=f"p50_budget_{b:.0f}",
                total_cost=p50.total_cost,
                labor_cost=p50.labor_cost,
                understaff_cost=p50.understaff_cost,
                overstaff_cost=p50.overstaff_cost,
                budget_hours=b,
            )
        )
    return out
