# RetailPulse Architecture

## System in one paragraph

Raw store-day sales are ingested, validated against data contracts, and turned
into leakage-safe features. A champion model (global LightGBM quantiles, with
seasonal-naive as the permanent regression gate) produces calibrated
P10/P50/P90 forecasts on rolling-origin backtest folds and a sealed final
holdout. Those forecasts feed an OR-Tools staffing optimizer that allocates a
limited labor budget across stores, compared against simpler policies in an
explicitly simulated cost model. FastAPI + Streamlit serve the results;
Prefect flows and DuckDB-backed monitoring close the loop.

## Key boundary

The forecasting core (`src/retailpulse/`) is deterministic and testable.
Prefect, FastAPI, Streamlit, and any cloud wrapper call that core — they do
not contain model logic. No feature is computed differently in training vs
serving (the train/serve skew contract).

## Pipeline

```text
download + future schedules -> validate -> Parquet + DuckDB
        -> leakage-safe features -> backtest + tune
        -> MLflow promotion gate -> P10/P50/P90 forecasts
        -> staffing optimizer -> FastAPI + dashboard
new actuals -> accuracy + drift monitoring -> backtest + tune
```

## Modules

| Module | Owns |
|---|---|
| `data/` | synthetic generator, ingestion, Pandera contracts, prediction-time availability table |
| `features/` | calendar/base features, leakage-safe lags and rolling stats |
| `evaluation/` | metric formulas, rolling-origin splitter, backtester, reports |
| `models/` | seasonal-naive/last-value baselines, classical benchmarks, LightGBM candidate, TFT/Chronos challengers, promotion gate, champion selection |
| `calibration/` | quantile-crossing repair, conformal corrections |
| `optimization/` | staffing simulator, OR-Tools optimizer, policy comparison |
| `monitoring/` | forecast-vs-actual persistence, rolling metrics, alerts |
| `api/` | FastAPI app + Pydantic schemas |
| `dashboard/` | Streamlit UI (API-only client) |
| `flows/` | Prefect flows (ingest, forecast, optimize, monitor, daily) |

## Prediction-time contract

- `static`: store metadata (known before the dataset begins)
- `scheduled`: calendar, holidays, planned promos, planned opening status
- `observed-late`: Customers (forbidden as a future feature)
- `target`: Sales (never a feature)

Feature builders assert against this table; the leakage test suite enforces it.
