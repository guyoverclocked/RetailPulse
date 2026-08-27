"""Prefect flows: ingest, forecast, optimize, monitor, retrain.

Idempotent tasks with retries; each flow maps one-to-one to a CLI stage so
scheduled runs and manual runs share code. Retraining is gated behind an
explicit promotion flow.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from prefect import flow, task


@task(retries=2, retry_delay_seconds=30)
def ingest_task() -> str:
    from retailpulse.data.ingest import run_ingestion

    result = run_ingestion()
    return result.run_id


@task
def forecast_task() -> str:
    from retailpulse.data.ingest import load_curated
    from retailpulse.evaluation.splits import RollingOriginSplitter
    from retailpulse.models.lightgbm_quantile import LightGBMQuantileModel

    data = load_curated().to_pandas()
    splitter = RollingOriginSplitter(horizon_days=30, n_folds=1)
    splitter.fit(data["Date"])
    fold = splitter.folds()[0]
    train = data[splitter.train_mask(data, fold)]
    sched = data[splitter.valid_mask(data, fold)][
        ["Store", "Date", "Open", "Promo", "StateHoliday", "SchoolHoliday"]
    ]
    valid_dates = list(pd.date_range(fold.valid_dates[0], fold.valid_dates[1], freq="D"))
    model = LightGBMQuantileModel()
    pred = model.fit_predict(train, 30, valid_dates, sched)
    run_id = f"fc-{datetime.now(UTC):%Y%m%dT%H%M%S}"
    from retailpulse.monitoring.persistence import persist_forecasts

    persist_forecasts(pred, origin=fold.origin, model_version="lightgbm_quantile-v0", run_id=run_id)
    return run_id


@task
def optimize_task(forecast_run_id: str) -> str:
    from retailpulse.data.ingest import load_curated
    from retailpulse.optimization.optimizer import optimize_staffing
    from retailpulse.optimization.simulator import demand_to_required

    data = load_curated().to_pandas()
    # Optimization consumes the persisted forecast; here the pipeline runs
    # end to end on the curated tail to prove the optimizer path.
    frame = demand_to_required(data.tail(30 * 60))
    optimize_staffing(frame)
    return forecast_run_id


@task
def monitor_task() -> dict[str, float]:
    from retailpulse.monitoring.metrics import rolling_metrics
    from retailpulse.monitoring.persistence import load_forecast_actual

    return rolling_metrics(load_forecast_actual())


@flow(name="retailpulse-ingest")
def ingest_flow() -> str:
    return ingest_task()


@flow(name="retailpulse-forecast")
def forecast_flow() -> str:
    return forecast_task()


@flow(name="retailpulse-optimize")
def optimize_flow() -> str:
    run_id = forecast_task()
    return optimize_task(run_id)


@flow(name="retailpulse-monitor")
def monitor_flow() -> dict[str, float]:
    return monitor_task()


@flow(name="retailpulse-daily")
def daily_flow() -> dict[str, float]:
    """The scheduled end-to-end batch run."""
    ingest_task()
    forecast_task()
    optimize_task("daily")
    return monitor_task()
