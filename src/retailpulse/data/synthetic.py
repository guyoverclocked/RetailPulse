"""Deterministic Rossmann-shaped synthetic data generator.

The fixture mirrors the real Rossmann schema exactly (same column names, same
dtypes, same structural rules: ``Open=0 => Sales=0``, promo effects only apply
when open, store metadata joins by ``Store`` id). The same Pandera schemas
validate synthetic and real data, so fixtures cannot drift from production
shapes.

Generates three artifacts:
- ``data/sample/`` — medium dataset for demos and ``make reproduce``
- ``tests/fixtures/sample/`` — tiny committed dataset for CI
- in-memory frames for unit tests

Properties embedded deliberately so downstream tests can assert them:
weekday seasonality, store-type level effects, promo lifts, holiday dips,
competition effects, random closures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from retailpulse.config import PROJECT_ROOT, ensure_dir, get_config

RNG = np.random.default_rng


@dataclass(frozen=True)
class SyntheticParams:
    """Parameters for one synthetic dataset generation."""

    seed: int
    n_stores: int
    start_date: str
    end_date: str
    store_types: tuple[str, ...]
    assortments: tuple[str, ...]
    promo_rate: float
    closure_rate: float
    school_holiday_rate: float
    state_holiday_rate: float
    competition_distance_scale: float
    promo2_rate: float


def _make_store_table(params: SyntheticParams, rng: np.random.Generator) -> pd.DataFrame:
    n = params.n_stores
    store_ids = np.arange(1, n + 1, dtype=np.int64)
    store_type = rng.choice(list(params.store_types), size=n)
    assortment = rng.choice(list(params.assortments), size=n)

    # ~20% of stores lack competition data (mirrors real Rossmann missingness).
    has_competition = rng.random(n) > 0.2
    distance = np.where(
        has_competition,
        np.round(rng.exponential(params.competition_distance_scale, size=n), 0),
        np.nan,
    )
    since_month = np.where(
        has_competition,
        rng.integers(1, 13, size=n),
        np.nan,
    )
    since_year = np.where(
        has_competition,
        rng.integers(1990, 2015, size=n),
        np.nan,
    )

    promo2 = rng.random(n) < params.promo2_rate
    promo2_week = np.where(promo2, rng.integers(1, 53, size=n), np.nan)
    promo2_year = np.where(promo2, rng.integers(2010, 2015, size=n), np.nan)
    intervals = rng.choice(
        ["Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov", "Mar,Jun,Sept,Dec"],
        size=n,
        replace=True,
    )
    promo_interval = np.where(promo2, intervals, "")

    return pd.DataFrame(
        {
            "Store": store_ids,
            "StoreType": store_type,
            "Assortment": assortment,
            "CompetitionDistance": distance,
            "CompetitionOpenSinceMonth": since_month,
            "CompetitionOpenSinceYear": since_year,
            "Promo2": promo2.astype(np.int64),
            "Promo2SinceWeek": promo2_week,
            "Promo2SinceYear": promo2_year,
            "PromoInterval": promo_interval,
        }
    )


def _german_state_holidays(start: str, end: str) -> pd.DatetimeIndex:
    """Fixed German holiday dates (Rossmann is a German chain).

    A static curated list keeps the generator deterministic and lets tests
    assert that holiday effects exist on exact dates.
    """
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    dates: list[pd.Timestamp] = []
    fixed = [
        (1, 1),  # New Year
        (5, 1),  # Labour Day
        (10, 3),  # German Unity Day
        (12, 25),  # Christmas Day 1
        (12, 26),  # Christmas Day 2
    ]
    for year in years:
        for month, day in fixed:
            dates.append(pd.Timestamp(year=year, month=month, day=day))
    return pd.DatetimeIndex(dates)


def _make_train_table(params: SyntheticParams, stores: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.date_range(params.start_date, params.end_date, freq="D")
    holidays = _german_state_holidays(params.start_date, params.end_date)
    holiday_set = set(holidays.date)

    frames: list[pd.DataFrame] = []
    for row in stores.itertuples(index=False):
        n = len(dates)
        dow = np.array([d.dayofweek for d in dates], dtype=np.int64)  # Mon=0..Sun=6

        # Base customer volume per store type, with persistent store noise.
        type_level = {"a": 900.0, "b": 700.0, "c": 450.0, "d": 300.0}[str(row.StoreType)]
        store_scale = rng.lognormal(mean=0.0, sigma=0.25)
        base = type_level * store_scale

        # Weekday seasonality: Mon high, Sun low (typical Rossmann pattern).
        dow_mult = np.array([1.15, 1.05, 1.0, 1.05, 1.1, 1.25, 0.8], dtype=np.float64)
        trend = 1.0 + 0.05 * (np.arange(n) / max(n, 1))  # mild yearly growth

        customers = base * dow_mult[dow] * trend
        customers += rng.normal(0, base * 0.12, size=n)
        customers = np.maximum(customers, 0.0)

        # Promotions: only on open days, ~+30% uplift.
        promo = rng.random(n) < params.promo_rate
        state_holiday = np.array([d.date() in holiday_set for d in dates], dtype=bool)
        school_holiday = rng.random(n) < params.school_holiday_rate

        # Closures: random days plus a store-specific probability of closing on
        # state holidays (structural zeros). The configured state_holiday_rate
        # is the share of stores that close on state holidays.
        open_flag = rng.random(n) > params.closure_rate
        closes_on_holiday = rng.random() < params.state_holiday_rate
        if closes_on_holiday:
            open_flag = open_flag & ~state_holiday

        # Holiday effect on customers when open: big dips.
        holiday_dip = np.where(state_holiday, 0.35, 1.0)
        school_mult = np.where(school_holiday & ~state_holiday, 1.1, 1.0)

        # Competition effect: nearer competition slightly reduces customers.
        comp = row.CompetitionDistance
        if np.isnan(comp):
            comp_mult = 1.0
        else:
            comp_mult = 1.0 - 0.08 * np.exp(-float(comp) / 2000.0)

        # Promo2 stores get a mild ongoing uplift in the announced months.
        promo2_mult = np.ones(n)
        if row.Promo2 == 1:
            months = str(row.PromoInterval).split(",")
            promo2_mult = np.where(
                [d.strftime("%b") in months for d in dates], 1.15, 1.0
            )

        customers = customers * holiday_dip * school_mult * comp_mult * promo2_mult
        promo_boost = np.where(promo & open_flag, 1.3, 1.0)
        customers = customers * promo_boost

        # Sales ~ customers with a store-level rate and noise.
        rate = rng.uniform(0.8, 1.2)
        sales = customers * rate + rng.normal(0, base * 0.06, size=n)
        sales = np.maximum(sales, 0.0)

        # Structural rule: closed => zero sales/customers.
        customers = np.where(open_flag, customers, 0.0)
        sales = np.where(open_flag, sales, 0.0)

        frames.append(
            pd.DataFrame(
                {
                    "Store": np.full(n, row.Store, dtype=np.int64),
                    "DayOfWeek": dow + 1,  # Rossmann encodes Mon=1..Sun=7
                    "Date": dates,
                    "Sales": np.round(sales, 0).astype(np.int64),
                    "Customers": np.round(customers, 0).astype(np.int64),
                    "Open": open_flag.astype(np.int64),
                    "Promo": promo.astype(np.int64),
                    "StateHoliday": np.where(state_holiday, "a", "0"),
                    "SchoolHoliday": school_holiday.astype(np.int64),
                }
            )
        )

    out = pd.concat(frames, ignore_index=True)
    # Sort by date then store — mirrors the real dataset's ordering.
    return out.sort_values(["Date", "Store"]).reset_index(drop=True)


def generate_frames(params: SyntheticParams | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic store and train frames in memory."""
    if params is None:
        cfg = get_config().data.synthetic
        params = SyntheticParams(
            seed=cfg.seed,
            n_stores=cfg.n_stores,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            store_types=tuple(cfg.store_types),
            assortments=tuple(cfg.assortments),
            promo_rate=cfg.promo_rate,
            closure_rate=cfg.closure_rate,
            school_holiday_rate=cfg.school_holiday_rate,
            state_holiday_rate=cfg.state_holiday_rate,
            competition_distance_scale=cfg.competition_distance_scale,
            promo2_rate=cfg.promo2_rate,
        )
    rng = RNG(params.seed)
    stores = _make_store_table(params, rng)
    train = _make_train_table(params, stores, rng)
    return stores, train


def _write_dir(out_dir, params: SyntheticParams) -> None:
    stores, train = generate_frames(params)
    ensure_dir(out_dir)
    stores.to_csv(out_dir / "store.csv", index=False)
    train.to_csv(out_dir / "train.csv", index=False)


def generate_sample_dataset() -> None:
    """Write the medium synthetic dataset under data/sample/."""
    cfg = get_config().data
    params = SyntheticParams(
        seed=cfg.synthetic.seed,
        n_stores=cfg.synthetic.n_stores,
        start_date=cfg.synthetic.start_date,
        end_date=cfg.synthetic.end_date,
        store_types=tuple(cfg.synthetic.store_types),
        assortments=tuple(cfg.synthetic.assortments),
        promo_rate=cfg.synthetic.promo_rate,
        closure_rate=cfg.synthetic.closure_rate,
        school_holiday_rate=cfg.synthetic.school_holiday_rate,
        state_holiday_rate=cfg.synthetic.state_holiday_rate,
        competition_distance_scale=cfg.synthetic.competition_distance_scale,
        promo2_rate=cfg.synthetic.promo2_rate,
    )
    _write_dir(PROJECT_ROOT / "data" / "sample", params)


def generate_ci_fixture() -> None:
    """Write the tiny committed fixture under tests/fixtures/sample/."""
    cfg = get_config().data
    params = SyntheticParams(
        seed=cfg.synthetic.seed,
        n_stores=cfg.fixture.n_stores,
        start_date=cfg.fixture.start_date,
        end_date=cfg.fixture.end_date,
        store_types=tuple(cfg.synthetic.store_types),
        assortments=tuple(cfg.synthetic.assortments),
        promo_rate=cfg.synthetic.promo_rate,
        closure_rate=cfg.synthetic.closure_rate,
        school_holiday_rate=cfg.synthetic.school_holiday_rate,
        state_holiday_rate=cfg.synthetic.state_holiday_rate,
        competition_distance_scale=cfg.synthetic.competition_distance_scale,
        promo2_rate=cfg.synthetic.promo2_rate,
    )
    _write_dir(PROJECT_ROOT / "tests" / "fixtures" / "sample", params)


if __name__ == "__main__":
    generate_sample_dataset()
    generate_ci_fixture()
