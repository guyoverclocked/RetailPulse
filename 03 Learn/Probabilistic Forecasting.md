---
type: concept
status: not-started
start: 2026-10-26
deadline: 2026-10-29
estimated_hours: 2.5
tags: [retailpulse, uncertainty, quantiles]
---

# Probabilistic forecasting

## Why

Staffing decisions need risk ranges, not false certainty. P10/P50/P90 communicate plausible low, central, and high demand.

## What to understand

- Quantile loss penalizes under- and over-prediction asymmetrically.
- Quantiles must be calibrated and should not cross.
- Coverage alone is insufficient; very wide intervals can cover everything.
- Calibration must be checked by store group and horizon.

## Build connection

Train LightGBM quantile objectives in [[LightGBM Quantile Model]]. Compare empirical P10–P90 coverage with the nominal 80%, width, and pinball loss.

## Alternatives

- Residual bootstrapping is a strong model-agnostic baseline.
- Conformal calibration can repair coverage later.
- Gaussian intervals are simple but impose a distributional assumption.

## Done when

You can explain calibration versus sharpness and diagnose quantile crossing.

Previous: [[Forecast Metrics]] · Next: [[LightGBM Quantile Model]]

