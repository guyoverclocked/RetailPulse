---
type: phase
status: planned
start: 2026-09-28
deadline: 2026-10-25
estimated_hours: 20-24
tags: [retailpulse, phase-2]
---
# Phase 2 - Evaluation and Baselines

## Why

This phase creates the court in which every later model must compete fairly.

## Sequence

1. [[Global vs Local Models]] — Sep 28-29
2. [[Time-Series Backtesting]] — Sep 30-Oct 7
3. [[Forecast Metrics]] — Oct 8-11
4. [[Baselines]] — Oct 12-18
5. [[Testing and CI]] foundation — Oct 19-22
6. Baseline report and recovery — Oct 23-25

## Deliverable

Reusable rolling-origin backtester, sealed final holdout, seasonal-naive results, and error slices.

## Exit gate

No future data enters training, all folds share the production horizon, and baseline results can be regenerated in CI on sample data.

Next: [[Phase 3 - Global ML and Uncertainty]].
