---
type: build
status: not-started
start: 2026-10-19
deadline: 2027-01-24
estimated_hours: 8
tags: [retailpulse, testing, ci]
---

# Testing and CI

## Why

Tests make the forecast trustworthy and the repository safe to change.

## Test pyramid

- Unit: transforms, metrics, quantile ordering, optimizer constraints.
- Data contract: schemas, keys, ranges, availability times.
- Leakage: lag features and folds never see future rows.
- Integration: fixture data through forecast and optimization.
- Smoke: API, dashboard, Docker image, and one orchestration flow.

GitHub Actions should run Ruff, mypy, pytest, and a Docker build on pull requests. Keep large model training out of normal CI; test with tiny fixtures and stub artifacts.

## Alternatives

Notebook-only checks are fast initially but cannot protect refactors. Full end-to-end cloud tests are valuable later but expensive and brittle.

## Done when

A clean clone passes CI and one intentionally introduced leakage bug is caught.

Start during [[Phase 2 - Evaluation and Baselines]] and improve it in every later node.

