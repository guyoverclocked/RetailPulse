# Data directory

This directory holds data that must never be committed to Git.

## How to obtain the real data

RetailPulse uses the **Rossmann Store Sales** competition dataset from Kaggle,
subject to Kaggle competition rules. Accepting those rules at download time is
required; the raw files must not be redistributed.

1. Accept the competition rules on the Kaggle dataset page
   (`rossmann-store-sales`).
2. Download `train.csv` and `store.csv` into `data/raw/`:

   ```bash
   # Option A: Kaggle CLI (after `pip install kaggle` and configuring credentials)
   kaggle competitions download -c rossmann-store-sales -p data/raw/
   unzip data/raw/rossmann-store-sales.zip -d data/raw/

   # Option B: manual download from the Kaggle website into data/raw/
   ```

3. Run `retailpulse ingest` (or `make reproduce`).

## What is stored where

| Path | Content | Committed? |
|---|---|---|
| `data/raw/` | Original downloaded CSVs | **Never** (gitignored) |
| `data/sample/` | Synthetic dataset from `make sample-data` | **Never** (gitignored) |
| `data/processed/` | Validated curated Parquet + manifest | **Never** (gitignored) |
| `data/features/` | Feature tables | **Never** (gitignored) |
| `tests/fixtures/sample/` | Tiny synthetic fixture for CI | **Yes** — this is the only committed data |

## What must never be committed

- Raw Kaggle competition files
- Any table with row-level real data
- Model artifacts containing real row-level examples

See `vault/01 Project/Data and License Boundaries.md` for the full policy.
