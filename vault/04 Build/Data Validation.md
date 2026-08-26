---
type: build
status: not-started
start: 2026-09-17
deadline: 2026-09-20
estimated_hours: 5
tags: [retailpulse, validation, data]
---

# Data validation

## Why

Silent data errors become model errors. Contracts make failure early and explainable.

## Build

- Pandera schemas for raw and curated tables.
- Pydantic models for configuration and later API payloads.
- Checks for uniqueness, date continuity, valid categories, nonnegative sales, and known store IDs.
- Explicit treatment of missing `Open` values and closed-store structural zeros.
- Feature-availability assertions from [[Prediction-Time Contract]].

Generate a compact validation report and fail the pipeline on hard violations.

## Alternatives

Great Expectations offers richer data documentation but adds machinery. Pandera fits a typed Python portfolio project; SQL checks can supplement cloud tables.

## Done when

Tests intentionally corrupt fixture data and each corruption fails with a useful message.

Previous: [[Prediction-Time Contract]] · Next: [[Exploratory Analysis]]

