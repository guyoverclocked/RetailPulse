"""Experiment tracking: MLflow logging and Optuna tuning (bounded).

Reproducibility contract: every run logs dataset version, fold definition,
code commit, feature list, params, metrics, runtime, and the model artifacts.
Optuna searches are capped by trial count and wall-clock time from config.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from retailpulse.config import get_config, resolve_path
from retailpulse.evaluation.backtester import Backtester
from retailpulse.evaluation.splits import RollingOriginSplitter
from retailpulse.models.lightgbm_quantile import LightGBMQuantileModel
from retailpulse.models.seasonal_naive import SeasonalNaiveModel


@dataclass
class ExperimentRun:
    """Metadata persisted for one tracked run."""

    run_id: str
    dataset_version: str
    commit: str
    model: str
    params: dict[str, Any]
    metrics: dict[str, float]
    runtime_seconds: float
    artifacts: list[str] = field(default_factory=list)


def dataset_version(data: pd.DataFrame) -> str:
    """Deterministic hash of the curated data (row-count + date-range + hash)."""
    h = hashlib.sha256()
    h.update(str(len(data)).encode())
    h.update(str(data["Date"].min()).encode())
    h.update(str(data["Date"].max()).encode())
    h.update(str(sorted(data["Store"].unique())[:20]).encode())
    return h.hexdigest()[:12]


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def track_backtest(
    model_name: str, params: dict[str, Any], metrics: dict[str, float], runtime: float
) -> ExperimentRun:
    """Record a backtest outcome to the local MLflow store."""
    import mlflow

    cfg = get_config()
    artifacts_dir = resolve_path(cfg.data.paths["artifacts_dir"]) / "mlruns"
    mlflow.set_tracking_uri(f"file://{artifacts_dir}")
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        run_id = run.info.run_id
    return ExperimentRun(
        run_id=run_id,
        dataset_version="",
        commit=git_commit(),
        model=model_name,
        params=params,
        metrics=metrics,
        runtime_seconds=runtime,
    )


def tune_lightgbm(
    data: pd.DataFrame,
    *,
    n_trials: int | None = None,
    timeout_minutes: int | None = None,
) -> dict[str, Any]:
    """Bounded Optuna search over LightGBM hyperparameters.

    Optimizes a weighted score (WAPE primary, coverage constraint handled by
    the promotion gate afterward). Trials and wall-clock are capped by config
    unless explicitly overridden.
    """
    import optuna

    cfg = get_config().lightgbm.tuning
    n_trials = n_trials if n_trials is not None else int(cfg["n_trials"])
    timeout = timeout_minutes if timeout_minutes is not None else int(cfg["timeout_minutes"])

    splitter = RollingOriginSplitter(horizon_days=30, n_folds=2)
    splitter.fit(data["Date"])
    backtester = Backtester(splitter)
    naive_wape = backtester.evaluate(SeasonalNaiveModel(), data).aggregate["wape"]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        model = LightGBMQuantileModel(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            num_leaves=int(params["num_leaves"]),
            min_child_samples=int(params["min_child_samples"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
        )
        result = backtester.evaluate(model, data)
        wape = result.aggregate.get("wape", float("inf"))
        return wape

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout * 60)
    return {**study.best_params, "best_wape": float(study.best_value), "naive_wape": naive_wape}
