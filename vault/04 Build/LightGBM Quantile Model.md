---
type: build
status: not-started
start: 2026-10-30
deadline: 2026-11-08
estimated_hours: 12
tags: [retailpulse, lightgbm, modeling]
---

# LightGBM quantile model

## Why

A global gradient-boosted model is fast, strong on tabular covariates, and scalable across stores.

## Build

1. Use MLForecast for lag, rolling, and date features.
2. Add store attributes, promotions, holidays, and store ID.
3. Fit separate P10/P50/P90 LightGBM objectives.
4. Generate features inside each backtest fold to prevent leakage.
5. Enforce closed-store zeros and repair rare quantile crossings.
6. Report accuracy, calibration, segment robustness, training time, and inference time.

Start with a small hand-set configuration. Tune only after the pipeline is correct.

## Alternatives

CatBoost handles categories elegantly; XGBoost is equally credible. LightGBM is chosen for speed and direct quantile objectives.

## Done when

The model reproducibly beats seasonal naive on the primary scorecard without leakage.

Previous: [[Probabilistic Forecasting]] · Next: [[Experiment Tracking]]

