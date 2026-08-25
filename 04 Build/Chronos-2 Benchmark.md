---
type: build
status: not-started
start: 2026-12-02
deadline: 2026-12-06
estimated_hours: 5
tags: [retailpulse, foundation-model, benchmark]
---

# Chronos-2 benchmark

## Why

A zero-shot foundation model shows awareness of current forecasting approaches while preserving disciplined comparison.

## Build

1. Select a reproducible stratified store subset: large/small, stable/volatile, promo-sensitive/less-sensitive.
2. Run zero-shot forecasts under a fixed compute budget.
3. Evaluate with the same horizons and metrics.
4. Record setup friction, latency, memory, calibration, and limitations.
5. Treat this as an ablation, not the default production model.

## Alternatives

TimesFM and Moirai are valid benchmarks. Choose one foundation model only; the purpose is evidence, not a model zoo.

## Done when

A compact table compares Chronos-2 with seasonal naive and LightGBM on the identical subset.

Previous: [[TFT Challenger]] · Next: [[Decision-Focused Forecasting]]

