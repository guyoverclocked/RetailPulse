"""OR-Tools CP-SAT staffing optimizer.

Allocates integer staff-hours per store-day to minimize the simulated cost
(labor + understaffing penalty + overstaffing penalty) subject to:

- closed stores forced to 0 hours
- open stores at least ``min_hours``, at most ``max_hours``
- chain-wide daily labor budget
- integer staffing

Returns the allocation, cost breakdown, and infeasibility diagnostics when
the constraints cannot be satisfied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from retailpulse.config import get_config


@dataclass
class AllocationResult:
    """Optimizer output: the allocation plus cost and diagnostics."""

    allocation: pd.DataFrame  # Store, Date, required_hours, scheduled_hours
    total_cost: float
    labor_cost: float
    understaff_cost: float
    overstaff_cost: float
    infeasible: bool = False
    diagnostics: str = ""
    label: str = "simulated"


def optimize_staffing(
    demand_frame: pd.DataFrame,
    *,
    demand_col: str = "required_hours",
    open_col: str = "Open",
    store_col: str = "Store",
    date_col: str = "Date",
    budget_hours: float | None = None,
    min_hours: float | None = None,
    max_hours: float | None = None,
    time_limit_seconds: float = 60.0,
) -> AllocationResult:
    """Run the CP-SAT staffing optimization.

    ``demand_frame`` must contain required-hours per store-day (from the
    simulator's ``demand_to_required``), the store id, the date, and the
    open flag.
    """
    from ortools.sat.python import cp_model

    cfg = get_config().staffing
    assumptions = cfg.assumptions
    constraints = cfg.constraints
    budget = budget_hours if budget_hours is not None else float(constraints["budget_hours"])
    min_h = min_hours if min_hours is not None else float(constraints["min_hours_per_open_store"])
    max_h = max_hours if max_hours is not None else float(constraints["max_hours_per_store"])
    wage = float(assumptions["wage_per_hour"])
    under_pen = float(assumptions["understaff_penalty_per_hour"])
    over_pen = float(assumptions["overstaff_penalty_per_hour"])

    df = demand_frame.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values([date_col, store_col]).reset_index(drop=True)
    required = df[demand_col].to_numpy(dtype=np.float64)
    open_flags = df[open_col].to_numpy(dtype=np.int64)

    model = cp_model.CpModel()
    # Scale by 2 to handle half-hour precision if needed; here integer hours.
    scale = 1
    n = len(df)
    cap = int(np.ceil(max_h * scale))
    min_cap = int(np.ceil(min_h * scale))
    budget_cap = int(budget * scale)
    sched = [model.new_int_var(0, cap, f"s_{i}") for i in range(n)]

    # Objective terms are linear in hours, so we accumulate via
    # intermediate variables for |required - sched| decomposition.
    under = [model.new_int_var(0, cap, f"u_{i}") for i in range(n)]
    over = [model.new_int_var(0, cap, f"o_{i}") for i in range(n)]
    labor_terms = []
    under_terms = []
    over_terms = []
    for i in range(n):
        req_scaled = round(required[i] * scale)
        # u_i >= req - sched, o_i >= sched - req
        model.add(under[i] >= req_scaled - sched[i])
        model.add(over[i] >= sched[i] - req_scaled)
        model.add(under[i] >= 0)
        model.add(over[i] >= 0)
        labor_terms.append(sched[i])
        under_terms.append(under[i])
        over_terms.append(over[i])

    # Closed stores: zero hours.
    for i in range(n):
        if open_flags[i] == 0:
            model.add(sched[i] == 0)

    # Open stores: min coverage.
    for i in range(n):
        if open_flags[i] == 1:
            model.add(sched[i] >= min_cap)

    # Chain-wide budget per date.
    for date_val in df[date_col].unique():
        idx = df.index[df[date_col] == date_val].tolist()
        model.add(sum(sched[i] for i in idx) <= budget_cap)

    # Objective: minimize cost (linear, integer).
    wage_scaled = round(wage * 100)  # cents-scale for integer coefficients
    under_scaled = round(under_pen * 100)
    over_scaled = round(over_pen * 100)
    objective = (
        wage_scaled * sum(labor_terms)
        + under_scaled * sum(under_terms)
        + over_scaled * sum(over_terms)
    )
    model.minimize(objective)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    status = solver.solve(model)

    if status in (cp_model.INFEASIBLE, cp_model.MODEL_INVALID):
        return AllocationResult(
            allocation=df[[store_col, date_col, demand_col]]
            .rename(columns={demand_col: "required_hours"})
            .assign(scheduled_hours=0),
            total_cost=float("inf"),
            labor_cost=0.0,
            understaff_cost=0.0,
            overstaff_cost=0.0,
            infeasible=True,
            diagnostics=f"solver status: {solver.status_name(status)}",
        )

    sched_vals = np.array([solver.value(v) for v in sched], dtype=np.float64) / scale
    out = df[[store_col, date_col, demand_col, open_col]].copy()
    out = out.rename(columns={demand_col: "required_hours"})
    out["scheduled_hours"] = sched_vals

    # Cost from the simulator for one consistent breakdown.
    from retailpulse.optimization.simulator import evaluate_allocation

    cost = evaluate_allocation(
        out["required_hours"].to_numpy(),
        out["scheduled_hours"].to_numpy(),
        wage=wage,
        under_penalty=under_pen,
        over_penalty=over_pen,
    )
    return AllocationResult(
        allocation=out.drop(columns=[open_col]),
        total_cost=cost.total_cost,
        labor_cost=cost.labor_cost,
        understaff_cost=cost.understaff_cost,
        overstaff_cost=cost.overstaff_cost,
        diagnostics=f"solver status: {solver.status_name(status)}",
    )
