---
type: project
status: planned
start: 2026-09-04
deadline: 2026-09-05
estimated_hours: 1.5
tags: [retailpulse, metrics]
---
# Success Metrics

## Why

One accuracy number cannot prove forecast usefulness, calibration, or operability.

## Forecast evidence

- WAPE (primary), MAE, and MASE (secondary; RMSSE dropped — MASE's seasonal-naive scale suffices)
- bias by store segment and horizon
- P10/P50/P90 pinball loss
- P10-P90 empirical coverage and interval width
- skill versus seasonal naive

## Decision evidence

- simulated under-capacity store-days
- excess labor hours
- total modeled staffing cost
- sensitivity to productivity and penalty assumptions

## Engineering evidence

- clean-clone reproduction time
- test count and coverage for critical modules
- full batch runtime
- API p50/p95 latency
- data freshness, model age, and failed-run count

## Guardrail

Never describe simulated savings as realized business savings.

## Done when

Every future resume claim has a named metric, comparator, population, and evaluation period.

Learn more: [[Forecast Metrics]], [[Probabilistic Forecasting]], [[Monitoring Forecasts]].
