---
type: build
status: not-started
start: 2026-10-12
deadline: 2026-10-18
estimated_hours: 8
tags: [retailpulse, baseline, modeling]
---

# Baselines

## Why

Complex models have no value unless they beat credible simple forecasts.

## Build in order

1. Last value and seasonal naive at lag 7.
2. AutoETS and AutoARIMA through StatsForecast.
3. SARIMAX on selected representative stores when promo/holiday covariates justify it.
4. Residual-based P10/P90 intervals as an uncertainty baseline.

Run every candidate through the same [[Time-Series Backtesting]] and [[Forecast Metrics]] code. Force forecasts to zero when a store is known closed.

## Alternatives

Prophet is accessible but does not add a distinct lesson here. Classical models per store are benchmarks, not the scalable production path.

## Done when

The results table includes runtime, WAPE, bias, coverage, and failure count; seasonal naive remains a permanent regression gate.

Previous: [[Forecast Metrics]] · Next: [[Probabilistic Forecasting]]

