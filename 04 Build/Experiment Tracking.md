---
type: build
status: not-started
start: 2026-11-09
deadline: 2026-11-22
estimated_hours: 8
tags: [retailpulse, mlflow, optuna]
---

# Experiment tracking and tuning

## Why

Model selection must be reproducible, not based on notebook memory.

## Build

- Log dataset version, fold definition, code commit, features, parameters, metrics, runtime, and artifacts to MLflow.
- Use Optuna for a bounded LightGBM search.
- Optimize a weighted score led by WAPE or pinball loss; keep calibration as a promotion constraint.
- Save the best configuration, feature importance, and forecast samples.
- Register only candidates that pass validation and baseline gates.

Cap trials and runtime before starting. Tuning cannot rescue leakage or poor evaluation design.

## Alternatives

Weights & Biases offers polished hosted tracking. Plain files are simpler but make comparisons and lineage weaker. Local MLflow keeps the project self-contained.

## Done when

Another person can locate the winning run and reproduce its data, parameters, and metrics.

Previous: [[LightGBM Quantile Model]] · Next: [[Champion-Challenger Selection]]

