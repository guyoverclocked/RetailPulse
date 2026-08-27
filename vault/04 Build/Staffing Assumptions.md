---
type: build
status: not-started
start: 2026-12-17
deadline: 2026-12-19
estimated_hours: 2
tags: [retailpulse, phase-5, staffing]
---
# Staffing Assumptions

## Why

The optimizer can only convert forecast distributions into a defensible staffing
plan if every conversion rate and penalty is explicit, editable, and labeled as a
simulation.

## Cost model

For each store-day, required staff-hours:

```
h_req = base_hours_per_open_store + forecast_sales / labor_productivity_sales_per_hour   (0 for closed stores)
```

Cost of an allocation:

```
Σ wage_per_hour · h_sched
+ understaff_penalty_per_hour · Σ max(0, h_req - h_sched)
+ overstaff_penalty_per_hour  · Σ max(0, h_sched - h_req)
```

## Defaults (in `configs/staffing.yaml`, all editable)

| Parameter | Default | Meaning |
|---|---|---|
| `labor_productivity_sales_per_hour` | 100 | one staff-hour serves ~100 sales units |
| `wage_per_hour` | 15.0 | simulated wage (currency-neutral) |
| `base_hours_per_open_store` | 4.0 | minimum coverage even at zero forecast |
| `understaff_penalty_per_hour` | 45.0 | 3× wage: cost of lost service |
| `overstaff_penalty_per_hour` | 22.5 | 1.5× wage: idle labor |
| `horizon_days` | 30 | planning horizon |

## Constraints

- Integer staffing per store-day.
- `h_sched ≥ min_hours` when the store is open; `h_sched ≤ max_hours`.
- Chain budget: `Σ h_sched ≤ budget`.
- Closed stores forced to 0 hours.

## Scenario modes

- **P50 mode:** expected plan — P50 forecasts as workload.
- **P90 mode:** risk-averse plan — P90 forecasts as workload.

## Sensitivity

Report sweeps productivity ±50% and penalties ±50%. Every output in UI and docs
must be labeled **simulated cost model, not realized savings** (see
[[Success Metrics]] guardrail).

## Done when

The optimizer reads all constants from config, the sensitivity report exists, and
no simulated number is presented as realized savings.

Previous: [[Decision-Focused Forecasting]] · Next: [[Staffing Optimizer]]
