"""End-to-end reproduction entry point.

Runs the full pipeline on synthetic data: ingest -> validate -> backtest the
baselines -> write a report. Models grow into this path as later phases land.
"""

from __future__ import annotations

from pathlib import Path

from retailpulse.config import get_config, resolve_path
from retailpulse.data.ingest import load_curated, run_ingestion
from retailpulse.evaluation.backtester import Backtester
from retailpulse.evaluation.report import format_results
from retailpulse.evaluation.splits import RollingOriginSplitter
from retailpulse.models.seasonal_naive import LastValueModel, SeasonalNaiveModel


def run_reproduce() -> None:
    """Full reproduction on synthetic data."""
    cfg = get_config()
    run_ingestion()
    data = load_curated().to_pandas()

    splitter = RollingOriginSplitter(
        horizon_days=cfg.backtest.horizon_days,
        n_folds=cfg.backtest.n_folds,
    )
    splitter.fit(data["Date"])
    backtester = Backtester(splitter)

    naive = SeasonalNaiveModel()
    last_value = LastValueModel()
    results = [
        backtester.evaluate(naive, data),
        backtester.evaluate(last_value, data, naive=naive),
    ]

    report = format_results(results)
    reports_dir = resolve_path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path: Path = reports_dir / "baseline_report.md"
    out_path.write_text(f"# Baseline report (synthetic data)\n\n{report}\n")
    print(report)
    print(f"\nReport written to {out_path}")
