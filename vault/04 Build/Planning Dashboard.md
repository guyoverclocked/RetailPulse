---
type: build
status: not-started
start: 2027-01-09
deadline: 2027-01-13
estimated_hours: 7
tags: [retailpulse, streamlit, product]
---

# Planning dashboard

## Why

The dashboard lets a non-ML user inspect risk and make a planning choice.

## Build

- Portfolio overview: demand, coverage, bias, freshness.
- Store drill-down: actual history and P10/P50/P90 forecast band.
- Scenario controls: labor budget, penalty weights, P50/P90 risk mode.
- Staffing result: allocation, constraints, cost components, baseline comparison.
- Clear labels for simulated assumptions and stale data.

Call the API instead of importing model internals. Keep the main user journey under three screens.

## Alternatives

React offers greater UI control but shifts attention from forecasting. Streamlit plus Plotly is fast, inspectable, and enough for a strong demo.

## Done when

A first-time user can select a store, understand uncertainty, run a scenario, and explain the result without reading code.

Previous: [[Forecast API]] · Next: [[Workflow Orchestration]]

