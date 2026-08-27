"""Pandera data contracts for raw and curated tables.

The same schemas validate synthetic fixtures and real Rossmann data, so
fixtures cannot drift from production shapes. Validation fails hard with a
precise message pointing at the violating rows/columns.
"""

from __future__ import annotations

import pandera.pandas as pa
import polars as pl

from retailpulse.data.availability import AVAILABILITY

# Raw Rossmann column sets (exact names from the competition files).
TRAIN_COLUMNS = [
    "Store",
    "DayOfWeek",
    "Date",
    "Sales",
    "Customers",
    "Open",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
]

STORE_COLUMNS = [
    "Store",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "Promo2",
    "Promo2SinceWeek",
    "Promo2SinceYear",
    "PromoInterval",
]


def train_schema() -> pa.DataFrameSchema:
    """Raw train-table contract (Rossmann ``train.csv`` shape)."""
    return pa.DataFrameSchema(
        {
            "Store": pa.Column(int, pa.Check.greater_than_or_equal_to(1)),
            "DayOfWeek": pa.Column(int, pa.Check.isin([1, 2, 3, 4, 5, 6, 7])),
            "Date": pa.Column(pa.DateTime, nullable=False),
            "Sales": pa.Column(int, pa.Check.greater_than_or_equal_to(0)),
            "Customers": pa.Column(int, pa.Check.greater_than_or_equal_to(0)),
            "Open": pa.Column(int, pa.Check.isin([0, 1])),
            "Promo": pa.Column(int, pa.Check.isin([0, 1])),
            "StateHoliday": pa.Column(str, pa.Check.isin(["0", "a", "b", "c"])),
            "SchoolHoliday": pa.Column(int, pa.Check.isin([0, 1])),
        },
        strict=True,
        coerce=True,
        unique=["Store", "Date"],
    )


def store_schema() -> pa.DataFrameSchema:
    """Raw store-table contract (Rossmann ``store.csv`` shape)."""
    return pa.DataFrameSchema(
        {
            "Store": pa.Column(int, pa.Check.greater_than_or_equal_to(1)),
            "StoreType": pa.Column(str, pa.Check.isin(["a", "b", "c", "d"])),
            "Assortment": pa.Column(str, pa.Check.isin(["a", "b", "c"])),
            "CompetitionDistance": pa.Column(float, nullable=True),
            "CompetitionOpenSinceMonth": pa.Column(float, nullable=True),
            "CompetitionOpenSinceYear": pa.Column(float, nullable=True),
            "Promo2": pa.Column(int, pa.Check.isin([0, 1])),
            "Promo2SinceWeek": pa.Column(float, nullable=True),
            "Promo2SinceYear": pa.Column(float, nullable=True),
            "PromoInterval": pa.Column(str, nullable=True),
        },
        strict=True,
        coerce=True,
        unique=["Store"],
    )


# Curated schema: the joined, typed table every downstream stage consumes.
def curated_schema() -> pa.DataFrameSchema:
    """Curated store-day table contract (post-join, post-clean)."""
    return pa.DataFrameSchema(
        {
            "Store": pa.Column(int, pa.Check.greater_than_or_equal_to(1)),
            "Date": pa.Column(pa.DateTime, nullable=False),
            "DayOfWeek": pa.Column(int, pa.Check.isin([1, 2, 3, 4, 5, 6, 7])),
            "Sales": pa.Column(int, pa.Check.greater_than_or_equal_to(0)),
            "Customers": pa.Column(int, pa.Check.greater_than_or_equal_to(0)),
            "Open": pa.Column(int, pa.Check.isin([0, 1])),
            "Promo": pa.Column(int, pa.Check.isin([0, 1])),
            "StateHoliday": pa.Column(str, pa.Check.isin(["0", "a", "b", "c"])),
            "SchoolHoliday": pa.Column(int, pa.Check.isin([0, 1])),
            "StoreType": pa.Column(str, pa.Check.isin(["a", "b", "c", "d"])),
            "Assortment": pa.Column(str, pa.Check.isin(["a", "b", "c"])),
            "CompetitionDistance": pa.Column(float, nullable=True),
            "CompetitionOpenSinceMonth": pa.Column(float, nullable=True),
            "CompetitionOpenSinceYear": pa.Column(float, nullable=True),
            "Promo2": pa.Column(int, pa.Check.isin([0, 1])),
            "Promo2SinceWeek": pa.Column(float, nullable=True),
            "Promo2SinceYear": pa.Column(float, nullable=True),
            "PromoInterval": pa.Column(str, nullable=True),
        },
        strict=True,
        coerce=True,
        unique=["Store", "Date"],
    )


def validate_train(df: pl.DataFrame) -> pl.DataFrame:
    """Validate raw train frame (converts to pandas for pandera, back after)."""
    import pandas as pd

    validated = train_schema().validate(df.to_pandas())
    return pl.from_pandas(validated)


def validate_store(df: pl.DataFrame) -> pl.DataFrame:
    """Validate raw store frame."""
    import pandas as pd

    validated = store_schema().validate(df.to_pandas())
    return pl.from_pandas(validated)


def validate_curated(df: pl.DataFrame) -> pl.DataFrame:
    """Validate curated frame plus the structural-zero rule."""
    import pandas as pd

    validated = curated_schema().validate(df.to_pandas())
    # Structural rule: closed stores have zero sales (and zero customers).
    mask = (validated["Open"] == 0) & ((validated["Sales"] != 0) | (validated["Customers"] != 0))
    if mask.any():
        n = int(mask.sum())
        raise ValueError(f"structural-zero violation: {n} closed store-days have non-zero sales/customers")
    return pl.from_pandas(validated)


def all_columns_declared() -> list[str]:
    """Columns present in the raw schemas but missing from the availability table."""
    declared = set(AVAILABILITY)
    present = set(TRAIN_COLUMNS) | set(STORE_COLUMNS)
    return sorted(present - declared)
