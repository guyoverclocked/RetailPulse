.PHONY: setup setup-challengers sample-data test lint format typecheck reproduce api dashboard demo clean

# One-command clean-clone setup (Python 3.12 via uv).
setup:
	uv sync

# Full setup including deep-learning challengers (torch, darts, chronos).
setup-challengers:
	uv sync --all-extras

# Regenerate the medium synthetic dataset under data/sample/.
sample-data:
	uv run retailpulse sample-data

# Full test suite on committed fixtures (unit + data-contract + leakage + integration).
test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src

# End-to-end reproduction on synthetic data: ingest -> validate -> backtest -> forecast -> optimize.
reproduce:
	uv run retailpulse reproduce

api:
	uv run uvicorn app.api_main:app --reload --port 8000

dashboard:
	uv run streamlit run app/dashboard_main.py

# Local full stack: API + dashboard + scheduled flow via docker compose.
demo:
	docker compose up --build

clean:
	rm -rf artifacts reports data/processed data/features
