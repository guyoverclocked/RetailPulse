---
type: build
status: not-started
start: 2026-12-17
deadline: 2027-01-03
estimated_hours: 12
tags: [retailpulse, optimization, ortools]
---

# Staffing optimizer

## Why

This node turns forecasts into a constrained decision—the clearest proof that the project is end to end.

## Build

1. Define a transparent simulated conversion from sales demand to labor workload.
2. Define integer staffing or shift variables in OR-Tools CP-SAT.
3. Add minimum coverage, maximum staffing, budget, and availability constraints.
4. Minimize labor plus over/understaffing penalties.
5. Compare P50 and P90 plans with a fixed staffing rule.
6. Return allocation, cost breakdown, slack, and infeasibility diagnostics.

Keep assumptions in configuration and run sensitivity analysis. Never present simulated savings as real company savings.

## Alternatives

PuLP is simpler for linear models; CP-SAT better represents discrete shifts. A heuristic is a useful fallback and comparison.

## Done when

Tests verify every constraint and the API can explain why the chosen plan differs from the baseline.

Previous: [[Decision-Focused Forecasting]] · Next: [[Forecast API]]

