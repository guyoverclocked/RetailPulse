---
type: concept
status: not-started
start: 2027-01-17
deadline: 2027-01-18
estimated_hours: 1.5
tags: [retailpulse, monitoring, mlops]
---

# Monitoring forecasts

## Why

Forecast quality is observed only after actual sales arrive. Monitoring must separate pipeline health from delayed model performance.

## Three layers

1. **Data:** freshness, schema, missingness, store coverage.
2. **Model:** rolling WAPE, bias, pinball loss, interval coverage and width.
3. **Service:** job success, API latency/errors, artifact age.

Define warning and failure thresholds from historical backtests. Retraining is triggered by sustained failure, not one noisy day.

## Alternatives

Evidently speeds up standard reports, while custom SQL is clearer for time-series horizon metrics. Use persisted forecast-versus-actual tables as the source of truth; add Evidently only where it helps.

## Done when

Each alert has a metric, threshold, evaluation window, owner, and response playbook.

Previous: [[Workflow Orchestration]] · Next: [[Monitoring Pipeline]]

