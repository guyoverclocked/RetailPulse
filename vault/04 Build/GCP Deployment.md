---
type: build
status: not-started
start: 2027-01-21
deadline: 2027-01-24
estimated_hours: 8
tags: [retailpulse, gcp, deployment]
---

# GCP deployment

## Why

A coherent cloud path demonstrates operational understanding while the local version remains reproducible.

## Deploy

- Cloud Run: API and dashboard containers.
- Cloud Run Jobs: batch forecasting and monitoring.
- Cloud Scheduler: trigger batch jobs.
- GCS: artifacts and immutable data snapshots.
- BigQuery: forecasts, actuals, metrics, and dashboard tables.
- Secret Manager: credentials; never bake them into images.

Set budgets, minimum instances to zero, retention rules, and a teardown guide. Deploy a thin working slice before scaling data.

## Alternatives

AWS and Azure are equally valid; one cloud told coherently is stronger than three shallow deployments. A local-only demo remains the fallback if cost or access blocks cloud work.

## Done when

CI builds an image, one scheduled job writes a forecast, the API reads it, and teardown steps are tested.

Previous: [[Monitoring Pipeline]] · Next: [[README and Demo]]

