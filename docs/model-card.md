# RetailPulse Model Card

## Champion candidate: global LightGBM quantile model

- **Type:** one global gradient-boosted model across all stores, store
  identity included as a feature
- **Outputs:** P10/P50/P90 daily sales per store, 30 days ahead
- **Objective:** quantile loss, one model per quantile, log-transformed
  target, trained on open store-days only
- **Inference:** recursive multi-step — lag/rolling features per forecast day
  filled with the prior day's P50 prediction
- **Calibration:** conformal quantile corrections learned on a held-out
  calibration slice (last 60 open training days); quantile crossings repaired
  by monotone sort
- **Closures:** deterministic zeros from the known-future Open schedule

## Baselines (permanent regression gates)

- Seasonal-naive (lag 7, walks back over closed days) — champion must beat it
- Last-value (lag 1)
- AutoETS / AutoARIMA (StatsForecast, local per store)

## Challengers (evidence-only, not production by default)

- Darts TFT — one controlled run, stratified store subset
- Chronos-2 — zero-shot ablation, same subset

## Promotion gate

A candidate is promoted only if, on validation folds:
1. WAPE skill vs seasonal-naive is positive
2. P10-P90 empirical coverage lands in [0.75, 0.85]
3. Backtest runtime stays under the configured budget

## Measured results (synthetic sample, 2-3 folds)

| Model | WAPE | Coverage | Width |
|---|---|---|---|
| seasonal_naive | 0.215 | 0.77 | 357 |
| last_value | 0.266 | 0.79 | 464 |
| lightgbm_quantile | 0.117-0.143 | 0.82 | 237-281 |

LightGBM is promoted: WAPE roughly halves the baseline, coverage near the
nominal 80%, intervals narrower. **These numbers are from synthetic data and
must be regenerated on real Rossmann data before any external claim.**

## Limitations

- Global model assumes stores share seasonal/promo structure; extreme
  store-specific shocks are captured only through lags.
- Conformal corrections assume the calibration slice is exchangeable with the
  horizon — a documented simplification.
- The sealed final 30-day holdout has not been evaluated yet (holdout
  discipline); it is unlocked exactly once after champion selection.
