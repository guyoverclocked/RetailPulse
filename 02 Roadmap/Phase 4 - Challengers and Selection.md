---
type: phase
status: planned
start: 2026-11-23
deadline: 2026-12-13
estimated_hours: 16-20
tags: [retailpulse, phase-4]
---
# Phase 4 - Challengers and Selection

## Why

TFT and Chronos demonstrate modern forecasting only when compared under the same evidence standard.

## Sequence

1. [[Champion-Challenger Selection]] rules — Nov 23
2. [[TFT Challenger]] — Nov 24-Dec 1
3. [[Chronos-2 Benchmark]] — Dec 2-6
4. Common-fold comparison and decision — Dec 7-13

## Deliverable

Accuracy, calibration, runtime, memory, and complexity comparison plus a signed model-selection ADR.

## Alternatives

If compute is limited, TFT receives one controlled run and Chronos uses a stratified subset. LightGBM may remain champion.

## Exit gate

The selected production model has a documented reason beyond novelty.

Next: [[Phase 5 - Staffing Decision Layer]].
