# RetailPulse

**Probabilistic multi-store demand forecasting and workforce planning — from raw Rossmann data to staffing decisions, end to end.**

> Given each store's promotion schedule, holiday calendar, and planned opening status,
> forecast daily sales for all 1,115 stores over the next 30 days, produce calibrated
> P10/P50/P90 forecasts, and convert that uncertainty into a staffing-capacity plan
> that minimizes simulated under-capacity and excess-labor costs.

## Status

🚧 **Under active construction.** The project currently exists as a
structured learning-and-planning vault; implementation follows the phased roadmap below.
This README will grow into the full project front page as phases land.

| Phase | Focus | Outcome | Status |
|---|---|---|---|
| 0 | Orientation and design | Frozen problem, scope, metrics, architecture | 🔄 in progress |
| 1 | Data foundation | Reproducible validated data and EDA | ⬜ |
| 2 | Evaluation and baselines | Leakage-safe backtester and baseline report | ⬜ |
| 3 | Global ML and uncertainty | LightGBM quantile candidate + calibrated intervals | ⬜ |
| 4 | Challengers and selection | TFT/Chronos evidence and champion decision | ⬜ |
| 5 | Staffing decision layer | Tested staffing simulator and policy comparison | ⬜ |
| 6 | Product and operations | API, dashboard, orchestration, monitoring, deployment | ⬜ |
| 7 | Portfolio release | Public-ready repository and resume story | ⬜ |

## What this system does

A regional retail operations manager gets, every night:

- **P10/P50/P90 daily sales forecasts** for each of 1,115 stores, 30 days ahead
- Chain-total reconciled forecasts and **under-forecast risk rankings**
- A **recommended staffing allocation** under a configurable labor budget
- **Scenario comparisons** (conservative / expected / high-demand plans)
- Monitoring: rolling accuracy, bias, interval coverage, drift

The forecast is not the product — the *staffing decision* it improves is.

## Approach in one paragraph

Raw store-day sales are ingested via script (never committed) and validated against explicit
data contracts. Features are engineered leakage-safe: lags and rolling statistics only ever see
data before the forecast origin, future `Customers` counts are banned as unavailable at
prediction time, and closed stores get a deterministic zero rule instead of being "predicted."
Models are compared on identical rolling-origin folds with a sealed final 30-day holdout:
seasonal-naïve → classical ETS/ARIMA → global LightGBM quantile model (champion candidate) →
TFT / Chronos challengers kept only if they justify their complexity. Calibrated quantile
forecasts feed an OR-Tools optimizer that allocates limited labor hours across stores, compared
against simpler policies in an explicitly *simulated* cost analysis. Everything ships with tests
(unit, data-contract, leakage), CI, Docker, API, dashboard, and forecast-vs-actual monitoring.

## Planned stack

Python 3.12 · Polars/pandas/NumPy · Parquet + DuckDB · Pandera + Pydantic · StatsForecast ·
MLForecast + LightGBM · Darts/PyTorch Forecasting (TFT challenger) · Chronos-2 (research benchmark) ·
Optuna + MLflow · OR-Tools · Prefect · FastAPI · Streamlit + Plotly · pytest/Ruff/mypy ·
uv + Docker · GitHub Actions · GCP (Cloud Run, Scheduler, GCS, BigQuery)

Full reasoning per tool: [`01 Project/Technology Choices.md`](./01%20Project/Technology%20Choices.md)

## Data

Rossmann Store Sales (Kaggle) — subject to Kaggle competition rules, so **raw files are not
redistributed here**. See [`01 Project/Data and License Boundaries.md`](./01%20Project/Data%20and%20License%20Boundaries.md);
download instructions will land in `data/README.md` during Phase 1.

## Repository layout

```text
RetailPulse/
├── 00 Start Here/    # Vault entry point: home, definition of done, study rhythm
├── 01 Project/       # Brief, scope, metrics, architecture, tech choices, data rules
├── 02 Roadmap/       # Phase-by-phase implementation plan
├── 03 Learn/         # Concept notes: backtesting, quantiles, leakage contract, metrics
├── 04 Build/         # Implementation notes per component
├── 05 Portfolio/     # README/demo plan, resume story, retrospective
├── 06 Tracking/      # Progress dashboard, weekly log, decision log, risk register
└── 07 Templates/     # Weekly review, experiment record, ADR templates
```

The Obsidian vault is the project's working brain; production code will live under
`src/retailpulse/` once Phase 1 starts. Notebooks stay exploratory clients of the package.

## Working agreement

- Cadence: phased plan with recovery checkpoints (see [Master Roadmap](./02%20Roadmap/Master%20Roadmap.md))
- Every decision and scope change recorded in [`06 Tracking/Decision Log.md`](./06%20Tracking/Decision%20Log.md)
- Completion requires the checklist in [`00 Start Here/Definition of Done.md`](./00%20Start%20Here/Definition%20of%20Done.md)
- Resume bullets use measured backtest results only — no numbers are claimed before they exist
