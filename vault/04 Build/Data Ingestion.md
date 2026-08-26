---
type: build
status: not-started
start: 2026-09-09
deadline: 2026-09-14
estimated_hours: 6
tags: [retailpulse, data]
---

# Data ingestion

## Why

Reproducible data lineage is the first part of an end-to-end system.

## Build

1. Download Rossmann data manually or through authenticated Kaggle tooling.
2. Keep raw files immutable and gitignored; store checksums and a manifest.
3. Parse dates and join store metadata with Polars.
4. Write partitioned Parquet and register queryable DuckDB views.
5. Log row counts, date ranges, schema version, and run ID.

Keep a small synthetic fixture in Git so tests and demos do not require Kaggle credentials.

## Alternatives

Pandas is simpler and required by some libraries; use it at library boundaries. BigQuery belongs in the deployment path, not the local MVP.

## Done when

One idempotent command rebuilds curated tables without modifying raw data.

Boundaries: [[Data and License Boundaries]] · Next: [[Prediction-Time Contract]]

