"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from retailpulse.config import PROJECT_ROOT

FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "sample"


@pytest.fixture(scope="session")
def fixture_store() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / "store.csv")


@pytest.fixture(scope="session")
def fixture_train() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / "train.csv", parse_dates=["Date"])
