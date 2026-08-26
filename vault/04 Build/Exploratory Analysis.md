---
type: build
status: not-started
start: 2026-09-21
deadline: 2026-09-27
estimated_hours: 6
tags: [retailpulse, eda]
---

# Exploratory analysis

## Why

EDA should change model design, not become a gallery of charts.

## Answer these questions

- What seasonal patterns exist by weekday, month, and store type?
- How do promotions and holidays align with demand?
- How frequent are closures, missing values, and outliers?
- Which stores have short, sparse, or unusually volatile histories?
- Are train and final-period feature distributions materially different?

Create one reproducible notebook or report, then write five modeling decisions it supports. Use only past information for rolling statistics.

## Alternatives

Interactive profiling tools are quick but often generic. Prefer a small set of Plotly charts tied to concrete hypotheses.

## Done when

Every retained chart has a one-sentence finding and an implementation consequence.

Previous: [[Data Validation]] · Next: [[Global vs Local Models]]

