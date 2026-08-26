---
type: concept
status: not-started
start: 2026-11-23
deadline: 2026-12-13
estimated_hours: 2
tags: [retailpulse, modeling, governance]
---

# Champion–challenger selection

## Why

A portfolio project is stronger when model choice follows evidence rather than novelty.

## Candidates

- Mandatory baseline: seasonal naive.
- Classical benchmark: AutoETS/AutoARIMA and SARIMAX where justified.
- Production candidate: global LightGBM quantiles.
- Deep challenger: Darts TFT.
- Foundation benchmark: Chronos-2 on a stratified subset.

## Promotion gate

A challenger is promoted only if it improves meaningful backtest metrics, calibration, and segment robustness enough to justify runtime, complexity, and serving cost. Evaluate the untouched holdout once, after the choice is locked.

## Alternatives

An ensemble can be robust, but adds latency and explanation burden. Use it only if its gain survives all folds and segments.

## Done when

An ADR records the winner, evidence, rejected options, and rollback model using [[ADR Template]].

Previous: [[Experiment Tracking]] · Next: [[TFT Challenger]] and [[Chronos-2 Benchmark]]

