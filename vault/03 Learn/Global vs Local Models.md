---
type: concept
status: not-started
start: 2026-09-28
deadline: 2026-09-29
estimated_hours: 1.5
tags: [retailpulse, modeling]
---

# Global vs local models

## Why

RetailPulse has 1,115 related series. The choice determines scalability and how information is shared.

## Comparison

| Approach | Strength | Limitation |
|---|---|---|
| Local model per store | Store-specific behavior | Expensive; weak on short histories |
| Global model | Learns across stores; one deployable artifact | Needs store identity and careful features |
| Clustered models | Middle ground | Adds clustering and routing complexity |

Use local classical models as interpretable benchmarks and one global LightGBM model as the production candidate.

## Alternatives

Separate store models are valid when stores differ radically or regulation requires isolation. Hierarchical reconciliation matters when forecasts must sum across formal regions; this dataset lacks a trustworthy hierarchy, so do not invent one.

## Chain-total forecast

The [[Project Brief]] requires a chain-total forecast. Decision: **bottom-up
aggregation** — sum the per-store forecasts to get the chain total. No
reconciliation is applied (there is no trustworthy hierarchy to reconcile
against). Logged as an ADR in [[Decision Log]].

## Done when

The model registry records whether each candidate is local or global and why.

Previous: [[Exploratory Analysis]] · Next: [[Time-Series Backtesting]]

