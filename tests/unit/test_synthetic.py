"""Tests for the synthetic data generator."""

from __future__ import annotations

import pandas as pd

from retailpulse.data.synthetic import SyntheticParams, generate_frames


def _params() -> SyntheticParams:
    return SyntheticParams(
        seed=42,
        n_stores=8,
        start_date="2014-01-01",
        end_date="2014-06-30",
        store_types=("a", "b", "c", "d"),
        assortments=("a", "b", "c"),
        promo_rate=0.15,
        closure_rate=0.05,
        school_holiday_rate=0.15,
        state_holiday_rate=0.5,
        competition_distance_scale=5000.0,
        promo2_rate=0.5,
    )


def test_deterministic() -> None:
    stores_a, train_a = generate_frames(_params())
    stores_b, train_b = generate_frames(_params())
    pd.testing.assert_frame_equal(stores_a, stores_b)
    pd.testing.assert_frame_equal(train_a, train_b)


def test_shape_and_keys() -> None:
    stores, train = generate_frames(_params())
    assert set(train["Store"].unique()) == set(stores["Store"].unique())
    assert train["Date"].nunique() == 181  # Jan 1 - Jun 30 2014


def test_structural_zero_rule() -> None:
    _, train = generate_frames(_params())
    closed = train[train["Open"] == 0]
    assert (closed["Sales"] == 0).all()
    assert (closed["Customers"] == 0).all()


def test_state_holiday_closure_rule() -> None:
    _, train = generate_frames(_params())
    # state_holiday_rate=0.5 => about half the stores close on Jan 1 (holiday).
    holiday = train[(train["StateHoliday"] == "a") & (train["Date"] == "2014-01-01")]
    n_closed = (holiday["Open"] == 0).sum()
    assert 0 < n_closed < holiday.shape[0]  # some closed, some open


def test_promo_uplift_on_open_days() -> None:
    """Promo is scheduled ahead of time (may land on closed days), but its
    sales uplift only materializes when the store is open."""
    _, train = generate_frames(_params())
    open_days = train[train["Open"] == 1]
    with_promo = open_days.loc[open_days["Promo"] == 1, "Sales"].mean()
    without_promo = open_days.loc[open_days["Promo"] == 0, "Sales"].mean()
    assert with_promo > without_promo


def test_dow_range() -> None:
    _, train = generate_frames(_params())
    assert set(train["DayOfWeek"].unique()) <= {1, 2, 3, 4, 5, 6, 7}
