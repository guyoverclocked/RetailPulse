"""Backtest reports: result tables and segment slices.

One report schema for every model so the comparison table in the champion
decision is always the same shape.
"""

from __future__ import annotations

import pandas as pd

from retailpulse.evaluation.backtester import BacktestResult


def results_table(results: list[BacktestResult]) -> pd.DataFrame:
    """Aggregate metrics for a list of models, one row per model."""
    rows = []
    for r in results:
        row = {"model": r.model, "n_folds": len([f for f in r.folds if not f.failed])}
        row.update(r.aggregate)
        row["failed_folds"] = sum(1 for f in r.folds if f.failed)
        rows.append(row)
    return pd.DataFrame(rows)


def fold_table(result: BacktestResult) -> pd.DataFrame:
    """Per-fold detail for one model."""
    rows = []
    for f in result.folds:
        row = {
            "fold": f.fold_id,
            "runtime_seconds": f.runtime_seconds,
            "n_train_rows": f.n_train_rows,
            "n_valid_rows": f.n_valid_rows,
            "failed": f.failed,
        }
        row.update(f.scorecard)
        rows.append(row)
    return pd.DataFrame(rows)


def format_results(results: list[BacktestResult]) -> str:
    """Human-readable markdown table for CLI/reports."""
    df = results_table(results)
    cols = [
        "model",
        "wape",
        "mae",
        "bias",
        "mase",
        "skill_vs_naive",
        "pinball_mean",
        "coverage",
        "interval_width",
        "failed_folds",
    ]
    cols = [c for c in cols if c in df.columns]
    return str(df[cols].to_markdown(index=False, floatfmt=".4f"))
