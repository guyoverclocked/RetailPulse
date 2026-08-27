# RetailPulse Data Card

## Dataset

Rossmann Store Sales (Kaggle competition `rossmann-store-sales`). Raw files
are subject to Kaggle competition rules and are **never committed**. See
`data/README.md` for acquisition instructions.

## Schema (raw)

`train.csv`: `Store, DayOfWeek, Date, Sales, Customers, Open, Promo,
StateHoliday, SchoolHoliday`

`store.csv`: `Store, StoreType, Assortment, CompetitionDistance,
CompetitionOpenSinceMonth, CompetitionOpenSinceYear, Promo2, Promo2SinceWeek,
Promo2SinceYear, PromoInterval`

## Structural rules

- `Open=0` implies `Sales=0` and `Customers=0` (structural zero, enforced by
  the Pandera contract)
- Each `(Store, Date)` pair is unique
- `Sales`, `Customers` are non-negative integers
- `DayOfWeek` in 1..7; `Open`/`Promo`/`SchoolHoliday` in {0,1};
  `StateHoliday` in {0, a, b, c}

## Feature availability

| Column | Availability | Used as feature? |
|---|---|---|
| Store metadata (all) | static | yes |
| Date/DayOfWeek/StateHoliday/SchoolHoliday/Promo/Open | scheduled | yes |
| Customers | observed-late | **no** (forbidden) |
| Sales | target | **no** |

## Provenance

- Download: Kaggle (credentials required, rules accepted at download time)
- Manifest: SHA-256 checksums, row counts, retrieval date per file
- Synthetic fixtures: deterministic (seed 42), same schemas as production
  data; CI runs on the tiny committed fixture, demos on `data/sample/`

## Limitations

- Synthetic data models the real dataset's structure but not its exact
  distributions; headline results must be regenerated on real data before
  any resume/demo claims.
- Competition metadata (competition open date, promo2 intervals) is imputed
  in synthetic data and may differ from real-world completeness.
