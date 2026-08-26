---
type: build
status: not-started
start: 2026-11-24
deadline: 2026-12-01
estimated_hours: 8
tags: [retailpulse, deep-learning, darts]
---

# TFT challenger

## Why

Temporal Fusion Transformer tests whether a sequence model earns its additional cost on multi-series, known-future covariates.

## Build

1. Use Darts, not a second overlapping deep-learning framework.
2. Train first on a stratified subset of stores.
3. Include static store attributes, known-future calendar/promo variables, and past target history.
4. Produce quantile forecasts with the same folds and scorecard.
5. Record training hardware, time, inference latency, and failure modes.

Set a compute budget and stopping rule before training.

## Alternatives

N-BEATS is simpler but uses covariates less naturally. PyTorch Forecasting is powerful, but supporting it alongside Darts adds portfolio breadth without depth.

## Done when

The challenger has a fair, reproducible comparison and a keep/reject decision—not necessarily a win.

Previous: [[Champion-Challenger Selection]] · Next: [[Chronos-2 Benchmark]]

