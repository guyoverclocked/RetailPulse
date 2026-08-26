---
type: concept
status: not-started
start: 2026-12-14
deadline: 2026-12-16
estimated_hours: 2
tags: [retailpulse, optimization, decisions]
---

# Decision-focused forecasting

## Why

Lower forecast error is useful only if it improves an action. RetailPulse converts demand uncertainty into a staffing recommendation.

## Decision model

Use predicted demand as workload, then minimize a transparent simulated cost:

`labor cost + understaffing penalty + overstaffing penalty`

Subject to store coverage, shift bounds, labor budget, and integer staffing. Run P50 for the expected plan and P90 for a risk-averse scenario.

## Boundaries

This is a planning simulation, not a claim about actual Rossmann labor productivity. State all assumed conversion rates and costs.

## Alternatives

- Rule-based staffing is easier but weak under shared budgets.
- Linear programming is enough for continuous hours; CP-SAT handles discrete shifts.
- Directly optimizing model loss for business cost is advanced follow-up work.

## Done when

Forecast and optimization outputs can be compared against a simple staffing rule on cost and constraint violations.

Previous: [[Champion-Challenger Selection]] · Next: [[Staffing Optimizer]]

