---
type: phase
status: planned
start: 2026-10-26
deadline: 2026-11-22
estimated_hours: 22-26
tags: [retailpulse, phase-3]
---
# Phase 3 - Global ML and Uncertainty

## Why

The primary candidate must learn across stores and express uncertainty, not merely minimize one aggregate point error.

## Sequence

1. [[Probabilistic Forecasting]] — Oct 26-29
2. [[LightGBM Quantile Model]] — Oct 30-Nov 8
3. [[Experiment Tracking]] — Nov 9-13
4. Optuna tuning within training windows — Nov 14-17
5. Calibration and segment analysis — Nov 18-22, inside the Nov 16-22 recovery window from [[Recovery and Scope Rules]]

## Deliverable

P10/P50/P90 LightGBM forecasts, tracked runs, feature evidence, calibration plots, and comparison with seasonal naive.

## Exit gate

The candidate beats or honestly fails to beat baseline, with interval coverage and runtime reported.

Next: [[Phase 4 - Challengers and Selection]].
