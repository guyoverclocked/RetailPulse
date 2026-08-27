"""Prediction-time feature-availability contract.

The single source of truth for what a forecast may legally see. Every feature
used by any model must be declared here with an availability class; feature
builders assert against this table, and the leakage test suite enforces it.

Classes:
- ``static``:          known before the dataset begins (store metadata)
- ``scheduled``:       known in advance of the day it describes (calendar,
                       holidays, planned promos, planned opening status)
- ``observed-late``:   only known after the day ends (Customers) — forbidden as
                       a future feature
- ``target``:          what we forecast (Sales) — forbidden as any feature
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

#: Availability class of every source column. Any column absent from this table
#: may not be used as a model feature.
AVAILABILITY: dict[str, str] = {
    # store metadata — static
    "Store": "static",
    "StoreType": "static",
    "Assortment": "static",
    "CompetitionDistance": "static",
    "CompetitionOpenSinceMonth": "static",
    "CompetitionOpenSinceYear": "static",
    "Promo2": "static",
    "Promo2SinceWeek": "static",
    "Promo2SinceYear": "static",
    "PromoInterval": "static",
    # calendar and schedules — known ahead of time
    "Date": "scheduled",
    "DayOfWeek": "scheduled",
    "StateHoliday": "scheduled",
    "SchoolHoliday": "scheduled",
    "Promo": "scheduled",
    "Open": "scheduled",
    # observed only after the fact
    "Customers": "observed-late",
    # the target — never a feature
    "Sales": "target",
}

#: Classes that may be used as features at any forecast origin.
_LEGAL_FEATURE_CLASSES = {"static", "scheduled"}


class AvailabilityError(ValueError):
    """Raised when a feature would not exist at forecast time."""


@dataclass(frozen=True)
class FeatureDeclaration:
    """One declared feature with its availability class."""

    name: str
    availability: str
    origin: str  # what generates it, e.g. "lag_7 of Sales"

    @property
    def legal(self) -> bool:
        return self.availability in _LEGAL_FEATURE_CLASSES


def availability_table() -> pd.DataFrame:
    """The committed feature-availability table (source columns)."""
    rows = [
        {"column": name, "availability": cls}
        for name, cls in AVAILABILITY.items()
    ]
    return pd.DataFrame(rows).sort_values(["availability", "column"]).reset_index(drop=True)


def assert_legal_features(features: list[str], *, context: str = "") -> None:
    """Fail hard if any feature is unavailable at forecast time.

    Args:
        features: Feature names to check.
        context: Where the check runs, for error messages.
    """
    illegal = [f for f in features if AVAILABILITY.get(f) not in _LEGAL_FEATURE_CLASSES]
    if illegal:
        detail = ", ".join(f"{f} ({AVAILABILITY.get(f, 'undeclared')})" for f in illegal)
        where = f" [{context}]" if context else ""
        raise AvailabilityError(f"forecast-time contract violation{where}: {detail}")


def is_legal_feature(name: str) -> bool:
    """True if ``name`` (a source column) may be used as a feature."""
    return AVAILABILITY.get(name) in _LEGAL_FEATURE_CLASSES


def all_legal_features_declared() -> bool:
    """True if every declared column has a recognized availability class."""
    return all(cls in _LEGAL_FEATURE_CLASSES | {"observed-late", "target"} for cls in AVAILABILITY.values())


def assert_no_future_rows(
    features: pd.DataFrame,
    origin: date,
    *,
    date_col: str = "Date",
    context: str = "",
) -> None:
    """Fail if any feature row is dated after the forecast origin.

    Guards the lag/rolling pipeline: every training feature row must exist at
    or before the origin.
    """
    if features.empty:
        return
    max_date = features[date_col].max().date()
    if max_date > origin:
        where = f" [{context}]" if context else ""
        raise AvailabilityError(
            f"future rows in features{where}: latest {max_date} > origin {origin}"
        )
