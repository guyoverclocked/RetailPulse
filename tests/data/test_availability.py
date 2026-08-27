"""Prediction-time availability contract tests."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from retailpulse.data.availability import (
    AVAILABILITY,
    AvailabilityError,
    all_legal_features_declared,
    assert_legal_features,
    assert_no_future_rows,
    availability_table,
)


def test_target_and_observed_late_are_forbidden_features() -> None:
    for col in ("Sales", "Customers"):
        with pytest.raises(AvailabilityError):
            assert_legal_features([col], context="test")


def test_scheduled_and_static_are_legal() -> None:
    assert_legal_features(["Store", "StoreType", "Date", "DayOfWeek", "Promo", "Open"], context="test")


def test_undeclared_column_rejected() -> None:
    with pytest.raises(AvailabilityError, match="undeclared"):
        assert_legal_features(["NotAColumn"], context="test")


def test_all_declared_columns_are_legal() -> None:
    assert all_legal_features_declared()


def test_availability_table_shape() -> None:
    table = availability_table()
    assert set(table.columns) == {"column", "availability"}
    assert len(table) == len(AVAILABILITY)


def test_no_future_rows_passes_on_clean_frame() -> None:
    df = pd.DataFrame({"Date": pd.to_datetime(["2020-01-01", "2020-01-02"])})
    assert_no_future_rows(df, origin=date(2020, 1, 2))


def test_no_future_rows_rejects_future() -> None:
    df = pd.DataFrame({"Date": pd.to_datetime(["2020-01-03"])})
    with pytest.raises(AvailabilityError, match="future rows"):
        assert_no_future_rows(df, origin=date(2020, 1, 2))
