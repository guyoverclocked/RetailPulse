---
type: concept
status: not-started
start: 2026-09-30
deadline: 2026-10-07
estimated_hours: 3
tags: [retailpulse, evaluation, backtesting]
---

# Time-series backtesting

## Why

Random train/test splits leak the future. Rolling-origin evaluation recreates repeated real forecasting decisions.

## RetailPulse design

- Keep the final 30 days untouched until model selection ends.
- Before that holdout, create at least three expanding-window folds.
- Forecast 30 days per fold.
- Fit preprocessing only on each fold's training period.
- Report aggregate and per-store results.

```text
train ─────────► validate 1
train ─────────────► validate 2
train ─────────────────► validate 3 | final holdout
```

## Alternatives

- Sliding windows adapt faster but discard history.
- One split is cheaper but gives a fragile estimate.
- More folds improve confidence at higher compute cost.

## Done when

One reusable splitter feeds every model, and a unit test proves no training timestamp crosses its forecast origin.

Previous: [[Global vs Local Models]] · Next: [[Forecast Metrics]] and [[Baselines]]

