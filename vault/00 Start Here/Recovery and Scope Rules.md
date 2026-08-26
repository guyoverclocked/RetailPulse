---
type: guide
status: planned
start: 2026-08-31
deadline: 2027-01-31
estimated_hours: 0.5
tags: [retailpulse, schedule, risk]
---
# Recovery and Scope Rules

## Why

Deadlines should guide decisions, not encourage rushed or misleading work.

## Recovery windows

- **21-27 September:** data and EDA catch-up
- **19-25 October:** evaluation integration
- **16-22 November:** modeling and calibration catch-up
- **28 December-3 January:** year-end recovery

## Cut in this order

1. Chronos full-dataset evaluation; keep a stratified subset.
2. TFT tuning breadth; keep one documented challenger run.
3. Evidently; retain custom forecast monitoring.
4. Prefect UI or advanced cloud automation.
5. Extra dashboard styling.

Never cut:

- seasonal-naive baseline
- leakage-safe rolling backtest
- untouched final holdout
- data and feature contracts
- tests for critical transformations
- honest limitations

## Done when

Every delay results in a documented scope choice in [[Decision Log]], not an invisible quality reduction.
