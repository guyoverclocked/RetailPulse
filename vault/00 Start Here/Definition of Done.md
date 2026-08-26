---
type: acceptance
status: planned
start: 2026-08-31
deadline: 2027-01-31
estimated_hours: 1
tags: [retailpulse, acceptance]
---
# Definition of Done

RetailPulse is complete only when:

- [ ] `make reproduce` works from a clean clone on sample data.
- [ ] Raw data is obtained through documented instructions, not redistributed improperly.
- [ ] Feature-availability and leakage tests pass.
- [ ] All models use identical rolling-origin folds.
- [ ] A seasonal-naive baseline is reported.
- [ ] The final 30-day holdout remained untouched until model selection ended.
- [ ] P10/P50/P90 forecasts have pinball loss and coverage results.
- [ ] A staffing optimizer is compared with a simple policy.
- [ ] FastAPI and the dashboard run in Docker.
- [ ] Forecast-versus-actual monitoring works on backfilled data.
- [ ] CI passes lint, type, unit, integration, data, and leakage checks.
- [ ] A deployed demo or reproducible local demo exists.
- [ ] README, data card, model card, assumptions, and limitations are current.
- [ ] Resume bullets use measured results only.

Related: [[Success Metrics]], [[Testing and CI]], [[README and Demo]].
