---
type: concept
status: not-started
start: 2026-10-08
deadline: 2026-10-11
estimated_hours: 2
tags: [retailpulse, evaluation, metrics]
---

# Forecast metrics

## Why

No single metric captures accuracy, bias, and uncertainty. A small scorecard prevents optimizing the wrong behavior.

## Scorecard

Let `y` = actual sales, `ŷ` = point forecast, `ŷ_q` = q-quantile forecast for
`q ∈ {0.10, 0.50, 0.90}`, `N` = store-day pairs, `I(·)` = indicator.

- **WAPE (primary point metric):** `WAPE = Σ|y - ŷ| / Σ|y|` — portfolio-level,
  immune to zero-sales blowups.
- **MAE:** `MAE = (1/N) Σ|y - ŷ|`
- **Bias:** `Bias = (1/N) Σ(ŷ - y)` — positive = over-forecasting.
- **MASE (secondary):** `MASE = MAE_model / MAE_seasonal_naive`, computed on the
  same folds. RMSSE is dropped: MASE's seasonal-naive scale is the simpler
  sufficient comparison.
- **Pinball loss:** `L_q(y, ŷ_q) = q·(y - ŷ_q)` when `y ≥ ŷ_q`, else
  `(1-q)·(ŷ_q - y)`; reported per q and averaged over the three quantiles.
  (Pinball at P50 is MAE/2 — a redundancy, not a free extra metric.)
- **Coverage:** `(1/N) Σ I(ŷ_0.10 ≤ y ≤ ŷ_0.90)`; target ≈ 80% (the nominal band).
- **Interval width:** `(1/N) Σ(ŷ_0.90 - ŷ_0.10)`; interpreted only alongside coverage.
- **Skill vs seasonal naive:** `1 - WAPE_model / WAPE_sn` (positive = better).

Calculate metrics overall, by store, horizon, promotion status, and open/closed status. Never let large stores hide poor small-store behavior. Closed stores are deterministic zeros — forecasts and metrics both respect that rule.

## Alternatives

RMSPE matches the original competition but behaves badly near zero. MAPE has the same weakness. Keep them only as secondary comparisons if needed.

## Done when

One metrics module produces the same schema for every backtest and supports [[Champion-Challenger Selection]].

Previous: [[Time-Series Backtesting]] · Next: [[Baselines]]

