"""Staffing simulator: the ground-truth cost evaluator.

Converts forecast (or actual) demand into required staff-hours and evaluates
any allocation under the explicit simulated cost model from
``configs/staffing.yaml``:

    h_req = base_hours + demand / productivity        (0 for closed stores)
    cost  = Σ wage·h_sched
          + under·Σ max(0, h_req - h_sched)
          + over ·Σ max(0, h_sched - h_req)

Every result is labeled simulated — never presented as realized savings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from retailpulse.config import get_config


@dataclass(frozen=True)
class StaffingResult:
    """Cost breakdown for one allocation evaluated against one demand scenario."""

    total_cost: float
    labor_cost: float
    understaff_cost: float
    overstaff_cost: float
    understaff_hours: float
    overstaff_hours: float
    n_store_days: int
    label: str = "simulated cost model"


def required_hours(
    demand: np.ndarray,
    open_mask: np.ndarray,
    *,
    productivity: float,
    base_hours: float,
) -> np.ndarray:
    """Required staff-hours per store-day from a demand vector."""
    req = np.where(open_mask, base_hours + np.asarray(demand) / productivity, 0.0)
    return np.maximum(req, 0.0)


def evaluate_allocation(
    required: np.ndarray,
    scheduled: np.ndarray,
    *,
    wage: float,
    under_penalty: float,
    over_penalty: float,
) -> StaffingResult:
    """Evaluate an allocation against required hours.

    Args:
        required: required hours per store-day.
        scheduled: scheduled hours per store-day (integer shifts).
    """
    req = np.asarray(required, dtype=np.float64)
    sched = np.asarray(scheduled, dtype=np.float64)
    if req.shape != sched.shape:
        raise ValueError(f"shape mismatch: required {req.shape} vs scheduled {sched.shape}")

    under = np.maximum(req - sched, 0.0)
    over = np.maximum(sched - req, 0.0)
    labor = wage * sched.sum()
    under_cost = under_penalty * under.sum()
    over_cost = over_penalty * over.sum()

    return StaffingResult(
        total_cost=float(labor + under_cost + over_cost),
        labor_cost=float(labor),
        understaff_cost=float(under_cost),
        overstaff_cost=float(over_cost),
        understaff_hours=float(under.sum()),
        overstaff_hours=float(over.sum()),
        n_store_days=len(req),
    )


def demand_to_required(
    demand_frame: pd.DataFrame,
    *,
    demand_col: str = "Sales",
    open_col: str = "Open",
) -> pd.DataFrame:
    """Convert a store-day frame (with demand + Open) to required hours.

    Reads the simulated assumptions from config.
    """
    cfg = get_config().staffing.assumptions
    df = demand_frame.copy()
    open_mask = df[open_col].to_numpy(dtype=np.int64) == 1
    req = required_hours(
        df[demand_col].to_numpy(dtype=np.float64),
        open_mask,
        productivity=float(cfg["labor_productivity_sales_per_hour"]),
        base_hours=float(cfg["base_hours_per_open_store"]),
    )
    df["required_hours"] = req
    return df
