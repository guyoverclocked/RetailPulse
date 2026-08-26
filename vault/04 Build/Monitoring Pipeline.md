---
type: build
status: not-started
start: 2027-01-18
deadline: 2027-01-20
estimated_hours: 5
tags: [retailpulse, monitoring, mlops]
---

# Monitoring pipeline

## Why

Persisted evidence makes drift and degradation diagnosable after deployment.

## Build

1. Persist each forecast with origin, target date, quantile, model version, and run ID.
2. Join actuals only when they become available.
3. Calculate rolling WAPE, bias, pinball loss, coverage, width, and freshness.
4. Segment by store group and forecast horizon.
5. Produce warning/failure states plus retrain recommendations.
6. Add Evidently reports only for useful standard checks.

Use historical backtests to seed thresholds and document that production thresholds need later calibration.

## Alternatives

Managed observability platforms reduce setup but obscure the core logic. SQL tables plus visible rules better demonstrate understanding.

## Done when

A deliberately stale or biased fixture triggers the expected alert and links to a response action.

Previous: [[Monitoring Forecasts]] · Next: [[GCP Deployment]]

