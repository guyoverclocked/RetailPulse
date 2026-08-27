"""Data-contract validation tests: corrupt fixtures must fail with useful messages."""

from __future__ import annotations

import polars as pl
import pytest
from pandera.errors import SchemaError, SchemaErrors

from retailpulse.data.validate import (
    validate_curated,
    validate_store,
    validate_train,
)


def _good_train() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Store": [1, 1, 2],
            "DayOfWeek": [1, 2, 3],
            "Date": ["2020-01-01", "2020-01-02", "2020-01-01"],
            "Sales": [100, 0, 50],
            "Customers": [80, 0, 40],
            "Open": [1, 0, 1],
            "Promo": [0, 0, 1],
            "StateHoliday": ["0", "0", "a"],
            "SchoolHoliday": [0, 0, 0],
        },
        schema_overrides={"Date": pl.Date},
    )


def _good_store() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Store": [1, 2],
            "StoreType": ["a", "b"],
            "Assortment": ["a", "c"],
            "CompetitionDistance": [100.0, None],
            "CompetitionOpenSinceMonth": [1.0, None],
            "CompetitionOpenSinceYear": [2000.0, None],
            "Promo2": [0, 1],
            "Promo2SinceWeek": [None, 5.0],
            "Promo2SinceYear": [None, 2010.0],
            "PromoInterval": [None, "Jan,Apr,Jul,Oct"],
        }
    )


def test_good_train_passes() -> None:
    validate_train(_good_train())


def test_good_store_passes() -> None:
    validate_store(_good_store())


def test_bad_day_of_week_fails() -> None:
    bad = _good_train().with_columns(pl.lit(9).alias("DayOfWeek"))
    with pytest.raises((SchemaError, SchemaErrors)):
        validate_train(bad)


def test_negative_sales_fails() -> None:
    bad = _good_train().with_columns(pl.lit(-1).alias("Sales"))
    with pytest.raises((SchemaError, SchemaErrors)):
        validate_train(bad)


def test_duplicate_store_date_fails() -> None:
    bad = pl.concat([_good_train(), _good_train().head(1)])
    with pytest.raises((SchemaError, SchemaErrors)):
        validate_train(bad)


def test_extra_column_fails_strict() -> None:
    bad = _good_train().with_columns(pl.lit(0).alias("SneakyColumn"))
    with pytest.raises((SchemaError, SchemaErrors)):
        validate_train(bad)


def test_closed_store_with_sales_fails() -> None:
    bad = _good_train().with_columns(
        pl.when(pl.col("Open") == 0).then(pl.lit(1)).otherwise(pl.col("Sales")).alias("Sales")
    )
    with pytest.raises(ValueError, match="structural-zero"):
        _curated(bad)


def _curated(train: pl.DataFrame) -> pl.DataFrame:
    joined = train.join(_good_store(), on="Store", how="left")
    return validate_curated(joined)


def test_good_curated_passes() -> None:
    _curated(_good_train())
