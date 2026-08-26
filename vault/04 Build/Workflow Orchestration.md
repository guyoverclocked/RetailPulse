---
type: build
status: not-started
start: 2027-01-14
deadline: 2027-01-17
estimated_hours: 5
tags: [retailpulse, prefect, orchestration]
---

# Workflow orchestration

## Why

End to end means the system can repeat its work reliably, not just run cells in order.

## Build

Create Prefect flows for:

1. ingest → validate → curate;
2. feature build → forecast → persist;
3. optimize → publish dashboard tables;
4. evaluate actuals → update monitoring;
5. retrain and promote only through an explicit gate.

Make tasks idempotent, retry transient failures, persist run metadata, and surface actionable failure messages. Keep schedules configurable.

## Alternatives

Airflow is common in larger teams; Dagster emphasizes data assets. Prefect has a lighter local-to-cloud learning curve. Plain cron is acceptable for a tiny system but teaches less about observability.

## Done when

One command runs the local workflow, and rerunning it does not duplicate outputs.

Previous: [[Planning Dashboard]] · Next: [[Monitoring Forecasts]]

