---
type: build
status: not-started
start: 2026-09-07
deadline: 2026-09-10
estimated_hours: 4
tags: [retailpulse, setup]
---

# Environment setup

## Why

A recruiter must be able to clone and run the project without reconstructing your laptop.

## Build

1. Create Python 3.12 project metadata with `uv` and `pyproject.toml`.
2. Use a `src/` package, `tests/`, `configs/`, `data/`, `artifacts/`, and `docs/`.
3. Configure Ruff, mypy, pytest, pre-commit, and environment variables.
4. Add `make setup`, `make test`, and `make demo` or equivalent documented commands.
5. Add Docker after one local pipeline works.

## Alternatives

Poetry and Conda are valid; `uv` is chosen for fast, simple locking. If Python 3.12 blocks a critical dependency, document a controlled 3.11 fallback rather than mixing versions.

## Done when

A clean environment installs, imports the package, and passes one smoke test.

Roadmap: [[Phase 1 - Data Foundation]] · Next: [[Data Ingestion]]

