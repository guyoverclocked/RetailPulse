# RetailPulse Backtesting Protocol

## Design

- **Rolling-origin, expanding window:** every fold trains on everything up to
  an origin and validates the next 30 days (the production horizon).
- **Identical folds for every candidate:** one splitter feeds every model; no
  model gets a friendlier fold.
- **Sealed final holdout:** the last 30 days stay locked until champion
  selection ends, then are evaluated exactly once.
- **Known-future inputs:** models may see only the scheduled columns (Open,
  Promo, StateHoliday, SchoolHoliday) for future dates. Sales and Customers
  are never available.
- **Fit-per-fold:** all preprocessing, scaling, and encoding are fit on each
  fold's training period only.

## Leakage enforcement

- No training row may be dated after its fold's origin.
- Lag/rolling features must reference pre-origin rows only.
- Feature availability classes (static / scheduled / observed-late / target)
  are asserted at build time.
- The leakage test suite includes a deliberately-leaking pipeline and must
  catch it.

## Scorecard

WAPE (primary), MAE, bias, MASE (vs seasonal-naive), pinball loss per
quantile, P10-P90 coverage (target ≈ 80%), interval width, skill vs naive.
Reported overall and by store, horizon, promotion status, and open/closed
status.

## Reproducibility

`make reproduce` runs ingest -> validate -> backtest -> report on synthetic
data from a clean clone. CI runs the same path on the committed fixture.
