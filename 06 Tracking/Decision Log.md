---
type: tracker
status: active
start: 2026-08-31
deadline: 2027-01-31
estimated_hours: 1
tags: [retailpulse, decisions]
---

# Decision log

Use [[ADR Template]] for decisions that affect architecture, evaluation, or scope.

| ID | Decision | Why | Revisit when |
|---|---|---|---|
| D-001 | Forecast daily sales for 1,115 stores, 30 days ahead | Matches dataset and supports planning | Data proves horizon unusable |
| D-002 | Keep final 30 days untouched | Honest final evaluation | Never during tuning |
| D-003 | Global LightGBM is production candidate | Strong, scalable tabular baseline | Challenger wins promotion gate |
| D-004 | Darts TFT is the only deep challenger | Depth over duplicate frameworks | Darts cannot support requirements |
| D-005 | Staffing is a labeled simulation | Dataset lacks real labor operations | Real staffing data becomes available |
| D-006 | GCP is the sole cloud narrative | Coherent deployment story | Target role requires another cloud |

Add links to evidence as the project develops. See [[Champion-Challenger Selection]].

