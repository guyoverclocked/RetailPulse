---
type: architecture
status: planned
start: 2026-09-03
deadline: 2026-09-04
estimated_hours: 1.5
tags: [retailpulse, architecture, stack]
---

# Technology choices

## Why

Each tool must own a clear responsibility. The stack is evidence of an end-to-end system, not a checklist to maximize.

## Core path

| Layer | Choice | Why / alternative |
|---|---|---|
| Language | Python 3.12 | 3.11 fallback only for dependency conflicts |
| Data | Polars, pandas, NumPy | Polars for transforms; pandas at library boundaries |
| Local storage | Parquet + DuckDB | Reproducible and serverless; SQLite is weaker for analytics |
| Contracts | Pandera + Pydantic | Tables versus configs/API payloads |
| Forecasts | StatsForecast + MLForecast/LightGBM | Credible classical baselines plus scalable global quantiles |
| Experiments | MLflow + Optuna | Lineage and bounded tuning; local-first |
| Decision | OR-Tools CP-SAT | Discrete, constrained staffing; PuLP is simpler linear alternative |
| Product | FastAPI + Streamlit/Plotly | Clean API boundary and rapid decision UI |
| Workflow | Prefect | Lighter local-to-cloud path than Airflow |
| Quality | pytest, Ruff, mypy, pre-commit | Automated correctness and maintainability |
| Packaging | uv, `pyproject.toml`, Docker | Clean-clone reproducibility |
| CI/CD | GitHub Actions | Test and build on pull request; deploy on merge |

## Conditional evidence

| Layer | Choice | Keep only if |
|---|---|---|
| Deep challenger | Darts TFT | Fair subset/full backtest fits the compute budget |
| Foundation benchmark | Chronos-2 | Zero-shot subset comparison is reproducible |
| Monitoring helper | Evidently | It adds value beyond persisted custom metrics |
| Cloud | GCP: Cloud Run, Jobs, Scheduler, GCS, BigQuery | The thin slice works within cost/access limits |

Do not use both Darts and PyTorch Forecasting. Do not add Kafka, Kubernetes, Spark, or a feature store without a demonstrated requirement.

## Done when

You can explain what each tool owns, its simplest alternative, and what would justify replacing it.

Previous: [[System Architecture]] · Next: [[Environment Setup]]
