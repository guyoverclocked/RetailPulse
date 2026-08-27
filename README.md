# RetailPulse

**Probabilistic multi-store demand forecasting and workforce planning — from
raw store-day data to staffing decisions, end to end.**

> Given each store's promotion schedule, holiday calendar, and planned opening
> status, forecast daily sales for every store over the next 30 days, produce
> calibrated P10/P50/P90 forecasts, and convert that uncertainty into a
> staffing-capacity plan that minimizes simulated under-capacity and
> excess-labor costs.

## Status

**In active construction — phases land in dependency order.** The repository
is a working system: data foundation, leakage-safe evaluation court, global
ML candidate, challenger scaffolding, staffing decision layer, API/dashboard,
orchestration, and monitoring are all implemented and tested. The final
sealed holdout remains locked until champion selection completes, per the
backtesting protocol.

| Layer | What exists | Evidence |
|---|---|---|
| Data foundation | Ingestion, Pandera contracts, prediction-time availability table, deterministic synthetic fixture | `make sample-data`, `retailpulse validate` |
| Evaluation court | Rolling-origin splitter, sealed holdout, metric formulas, backtester, leakage suite | `tests/leakage/`, `make reproduce` |
| Models | Seasonal-naive + last-value baselines, classical benchmarks, global LightGBM quantile candidate with conformal calibration, TFT/Chronos challengers, promotion gate | `tests/integration/test_lightgbm.py` |
| Decision layer | Simulated staffing cost model, OR-Tools optimizer, policy comparison | `tests/unit/test_optimizer.py` |
| Product | FastAPI, Streamlit dashboard, Prefect flows, DuckDB monitoring, Docker | `docker compose up`, `make demo` |

## What the system does

A regional retail operations manager gets, every night:

- **P10/P50/P90 daily sales forecasts** for each store, 30 days ahead
- Chain-total forecasts (bottom-up aggregation) and **risk rankings**
- A **recommended staffing allocation** under a configurable labor budget
- **Scenario comparisons** (conservative / expected / high-demand plans)
- Monitoring: rolling accuracy, bias, interval coverage, drift

The forecast is not the product — the *staffing decision* it improves is.

## Approach in one paragraph

Raw store-day sales are ingested (never committed) and validated against
explicit data contracts. Features are engineered leakage-safe: lags and
rolling statistics only see data before the forecast origin, future
`Customers` counts are banned as unavailable at prediction time, and closed
stores get a deterministic zero rule instead of being "predicted." Models are
compared on identical rolling-origin folds with a sealed final 30-day
holdout: seasonal-naive → classical ETS/ARIMA → global LightGBM quantile
model (champion candidate) → TFT / Chronos challengers kept only if they
justify their complexity. Calibrated quantile forecasts feed an OR-Tools
optimizer that allocates limited labor hours across stores, compared against
simpler policies in an explicitly *simulated* cost analysis. Everything
ships with tests (unit, data-contract, leakage), CI, Docker, API, dashboard,
and forecast-vs-actual monitoring.

## Current measured results

These numbers are **from the deterministic synthetic dataset** — they
demonstrate the pipeline works end to end and the promotion gate functions,
but they are *not* claims about real Rossmann data. Real-data numbers are
regenerated with `make reproduce` after `retailpulse ingest`.

| Model | WAPE | P10-P90 coverage | Interval width |
|---|---|---|---|
| seasonal_naive | 0.215 | 0.77 | 357 |
| last_value | 0.266 | 0.79 | 464 |
| lightgbm_quantile | 0.117 | 0.82 | 237 |

LightGBM passes the promotion gate (skill vs naive +0.30, coverage in
[0.75, 0.85]).

## Quickstart

```bash
make setup          # uv sync, Python 3.12
make sample-data    # deterministic synthetic dataset
make test           # unit + data-contract + leakage + integration
make reproduce      # ingest -> validate -> backtest -> report
make api            # uvicorn on :8000
make dashboard      # streamlit on :8501
make demo           # docker compose up (API + dashboard + flows)
```

See `docs/architecture.md`, `docs/data-card.md`, `docs/model-card.md`, and
`docs/backtesting-protocol.md`.

## Data

Rossmann Store Sales (Kaggle) — raw files are subject to Kaggle competition
rules and are never redistributed. `data/README.md` explains acquisition,
storage, and what must never be committed. The committed fixture under
`tests/fixtures/sample/` is synthetic and CI-safe.

## Stack

Python 3.12 · Polars/pandas/NumPy · Parquet + DuckDB · Pandera + Pydantic ·
StatsForecast · LightGBM (recursive quantile inference + conformal
calibration) · Darts/Chronos (optional challengers) · Optuna + MLflow ·
OR-Tools CP-SAT · Prefect · FastAPI · Streamlit + Plotly · pytest/Ruff/mypy ·
uv + Docker · GitHub Actions.

## Working agreement

- Phases and scope live in `vault/` (the project's planning brain);
  production code lives in `src/retailpulse/`.
- Every decision and scope change is recorded in
  [`06 Tracking/Decision Log.md`](./vault/06%20Tracking/Decision%20Log.md).
- Completion requires the checklist in
  [`00 Start Here/Definition of Done.md`](./vault/00%20Start%20Here/Definition%20of%20Done.md).
- Resume bullets use measured backtest results only — no numbers are claimed
  before they exist.
