"""Policy comparison integration: fixed rule vs P50 vs P90 plans."""

from __future__ import annotations

import pandas as pd
import pytest

from retailpulse.optimization.policies import compare_policies


def _forecast_actual(n_stores: int = 6, n_days: int = 14) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    actual_rows = []
    for store in range(1, n_stores + 1):
        for d in range(n_days):
            actual = float(200 * store + 50 * (d % 7))
            rows.append(
                {
                    "Store": store,
                    "Date": pd.Timestamp("2015-07-01") + pd.Timedelta(days=d),
                    "Open": 1,
                    "Sales_q50": actual * 0.9,
                    "Sales_q90": actual * 1.15,
                }
            )
            actual_rows.append(
                {
                    "Store": store,
                    "Date": pd.Timestamp("2015-07-01") + pd.Timedelta(days=d),
                    "Open": 1,
                    "Sales": actual,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(actual_rows)


def test_compare_policies_returns_all_three() -> None:
    forecast, actual = _forecast_actual()
    results = compare_policies(forecast, actual, budget_hours=500.0)
    names = {r.policy for r in results}
    assert names == {"fixed_rule", "p50_plan", "p90_plan"}
    for r in results:
        assert r.total_cost >= 0


def test_p90_plan_has_less_understaff_than_p50() -> None:
    """P90 uses a higher workload, so understaffing cost should not exceed P50's."""
    forecast, actual = _forecast_actual()
    results = compare_policies(forecast, actual, budget_hours=500.0)
    p50 = next(r for r in results if r.policy == "p50_plan")
    p90 = next(r for r in results if r.policy == "p90_plan")
    assert p90.understaff_cost <= p50.understaff_cost + 1e-6


def test_fixed_rule_is_independent_of_budget() -> None:
    forecast, actual = _forecast_actual()
    r1 = next(
        r
        for r in compare_policies(forecast, actual, budget_hours=100.0)
        if r.policy == "fixed_rule"
    )
    r2 = next(
        r
        for r in compare_policies(forecast, actual, budget_hours=1000.0)
        if r.policy == "fixed_rule"
    )
    assert r1.total_cost == pytest.approx(r2.total_cost)
