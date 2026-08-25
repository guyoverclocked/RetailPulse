---
type: concept
status: not-started
start: 2026-09-15
deadline: 2026-09-16
estimated_hours: 1.5
tags: [retailpulse, data, leakage]
---

# Prediction-time contract

## Why

A forecast is valid only if every feature would truly exist when the forecast is issued. Violating this rule produces impressive but unusable scores.

## Contract for RetailPulse

At forecast time, we may know store metadata, calendar fields, holidays, and planned promotions. We do **not** know future `Sales` or `Customers`. `Open=0` is a structural zero, not a normal low-sales day.

## Learn by doing

1. Label every source column as known, scheduled, observed-late, or target.
2. Record its availability time and owner.
3. Reject any feature whose timestamp is later than the forecast origin.
4. Add a leakage test for lag and rolling features.

## Alternatives

- Dropping all external variables is safer but wastes known-future information.
- Using forecasted customer counts is possible later, but creates a second model and error chain; it is outside the MVP.

## Done when

The feature-availability table is committed and [[Data Validation]] tests enforce it.

Previous: [[Data Ingestion]] · Next: [[Data Validation]]

