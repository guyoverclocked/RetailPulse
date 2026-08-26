---
type: build
status: not-started
start: 2027-01-04
deadline: 2027-01-08
estimated_hours: 7
tags: [retailpulse, fastapi, serving]
---

# Forecast API

## Why

An API separates modeling from its consumers and demonstrates a deployable contract.

## Build

- `GET /health`
- `GET /forecasts/{store_id}` with horizon and quantiles
- `POST /staffing/optimize` with budget and risk setting
- `GET /monitoring/summary`

Use Pydantic request/response models, version the routes, validate store IDs, and return model/data timestamps. Load a promoted artifact once at startup. Add structured logs and integration tests.

## Alternatives

Bentoml or KServe can manage model serving at larger scale. FastAPI is easier to understand and sufficient for this portfolio system.

## Done when

OpenAPI docs work, invalid inputs fail clearly, and Dockerized integration tests exercise forecast-to-optimization flow.

Previous: [[Staffing Optimizer]] · Next: [[Planning Dashboard]]

