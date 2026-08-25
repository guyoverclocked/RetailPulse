---
type: architecture
status: planned
start: 2026-09-02
deadline: 2026-09-04
estimated_hours: 2
tags: [retailpulse, architecture]
---
# System Architecture

## Why

The architecture prevents notebook-only logic and makes data, model, decision, and operating boundaries visible.

```mermaid
flowchart LR
    A[Download and future schedules] --> B[Validate]
    B --> C[Parquet and DuckDB]
    C --> D[Leakage-safe features]
    D --> E[Backtest and tune]
    E --> F[MLflow promotion gate]
    F --> G[P10 P50 P90 forecasts]
    G --> H[Staffing optimizer]
    G --> I[FastAPI and dashboard]
    H --> I
    J[New actuals] --> K[Accuracy and drift monitoring]
    K --> E
```

## Key boundary

The forecasting core is deterministic and testable. Prefect, FastAPI, Streamlit, and GCP call that core; they do not contain model logic.

## Alternatives

- **Notebook pipeline:** faster initially, weak to test and deploy.
- **Microservices:** excessive for one developer and one batch workflow.
- **Modular monolith:** selected; clear modules with one repository and deployable containers.

## Done when

Every box has one owning module and no feature is computed differently during training and serving.

Next: [[Technology Choices]], [[Build MOC]], and [[Prediction-Time Contract]].
