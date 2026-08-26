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

- **WAPE:** portfolio-level magnitude error; primary point metric.
- **MAE:** interpretable average absolute error.
- **Bias:** systematic over- or under-forecasting.
- **Pinball loss:** quality of P10/P50/P90 quantiles.
- **Coverage:** fraction of actuals inside P10–P90; target is near 80%.
- **Interval width:** sharpness, interpreted alongside coverage.

Calculate metrics overall, by store, horizon, promotion status, and open/closed status. Never let large stores hide poor small-store behavior.

## Alternatives

RMSPE matches the original competition but behaves badly near zero. MAPE has the same weakness. Keep them only as secondary comparisons if needed.

## Done when

One metrics module produces the same schema for every backtest and supports [[Champion-Challenger Selection]].

Previous: [[Time-Series Backtesting]] · Next: [[Baselines]]

