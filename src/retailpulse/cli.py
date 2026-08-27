"""RetailPulse command-line interface.

Every pipeline stage is one idempotent subcommand. Stages chain through the same
entry points the Prefect flows call, so CLI runs and scheduled runs share code.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="retailpulse",
    help="Probabilistic multi-store demand forecasting and workforce planning.",
    no_args_is_help=True,
)

# Imported lazily inside each command so `retailpulse --help` stays instant even
# with heavy ML dependencies installed.


@app.command()
def sample_data(fixture: bool = typer.Option(False, "--fixture", help="Also regenerate the committed CI fixture")) -> None:
    """Generate the medium synthetic dataset under data/sample/."""
    from retailpulse.data.synthetic import generate_ci_fixture, generate_sample_dataset

    generate_sample_dataset()
    typer.echo("Synthetic dataset written to data/sample/")
    if fixture:
        generate_ci_fixture()
        typer.echo("CI fixture regenerated under tests/fixtures/sample/")


@app.command()
def ingest() -> None:
    """Download/read raw Rossmann data and build validated curated tables."""
    from retailpulse.data.ingest import run_ingestion

    run_ingestion()


@app.command()
def validate() -> None:
    """Run data-contract validation on curated tables and fail hard on violations."""
    from retailpulse.data.validate import run_validation

    run_validation()


@app.command()
def reproduce() -> None:
    """End-to-end reproduction on synthetic data: ingest -> backtest -> optimize."""
    from retailpulse.cli_reproduce import run_reproduce

    run_reproduce()


if __name__ == "__main__":
    app()
