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
from retailpulse.models.lightgbm_quantile import LightGBMQuantileModel
from retailpulse.models.registry import evaluate_promotion
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

    naive_result = backtester.evaluate(naive, data)
    lgbm = LightGBMQuantileModel()
    lgbm_result = backtester.evaluate(lgbm, data, naive=naive)
    decision = evaluate_promotion(lgbm_result, naive_wape=naive_result.aggregate["wape"])
    results = [
        naive_result,
        backtester.evaluate(last_value, data, naive=naive),
        lgbm_result,
    ]

    report = format_results(results)
    reports_dir = resolve_path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path: Path = reports_dir / "baseline_report.md"
    out_path.write_text(
        f"# Baseline report (synthetic data)\n\n{report}\n\nPromotion gate: {decision.summary}\n"
    )
    print(report)
    print(f"\n{decision.summary}")
    print(f"\nReport written to {out_path}")
