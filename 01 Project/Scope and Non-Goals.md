---
type: project
status: planned
start: 2026-09-01
deadline: 2026-09-02
estimated_hours: 1
tags: [retailpulse, scope]
---
# Scope and Non-Goals

## In scope

- 30-day daily sales forecasts for all stores
- known future promotion, holiday, and opening information
- seasonal-naive, classical, global ML, and limited challenger comparison
- probabilistic forecasts and calibration
- simulated staffing allocation
- local reproducibility, API, dashboard, monitoring, CI, and one cloud deployment

## Not in scope

- SKU-level inventory or replenishment
- causal claims about promotion effectiveness
- real Rossmann staffing recommendations
- real-time streaming; this is scheduled batch forecasting
- customer forecasting using future `Customers`
- Kubernetes, Kafka, Spark, or multiple cloud providers
- production use of TFT or Chronos without evidence

## Why

Each non-goal prevents an unsupported claim or an infrastructure distraction.

## Done when

Every planned feature maps to [[Project Brief]] or is removed in [[Decision Log]].

Next: [[System Architecture]].
