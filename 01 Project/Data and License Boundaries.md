---
type: governance
status: planned
start: 2026-09-07
deadline: 2026-09-08
estimated_hours: 1.5
tags: [retailpulse, data, governance]
---
# Data and License Boundaries

## Why

The Rossmann competition data is subject to Kaggle competition rules. A portfolio repository must document access and avoid unapproved redistribution.

## Rules

- Do not commit raw competition files to the new public repository.
- Provide Kaggle CLI/download instructions requiring the user to accept the rules.
- Commit only schemas, metadata, a tiny synthetic sample, and derived aggregate examples when permitted.
- Record source URL, retrieval date, checksum, row count, and license/terms.
- Keep model artifacts and screenshots free of confidential or unnecessary row-level data.

## Alternatives

- **Public Apache-licensed mirror:** easier, but provenance may be weaker; verify before use.
- **Synthetic sample:** ideal for CI, not for headline results.
- **Different open dataset:** valid fallback if competition terms block publication.

## Done when

`data/README.md` explains how to obtain data, what is stored, and what must never be committed.

Next: [[Data Ingestion]] and [[Data Validation]].
