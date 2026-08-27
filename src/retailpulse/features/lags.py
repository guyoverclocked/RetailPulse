"""Leakage-safe lag and rolling features.

Every lag/rolling feature is built PER FOLD, only from rows dated strictly
before the forecast origin. A helper builds features for a training set that
already excludes the validation window; the leakage test suite then asserts
that no feature value on any validation row depends on future data.

Implementation: lags are computed within each store series via ``shift``, and
rolling stats via closed windows. Because the caller passes only pre-origin
rows, shifted/rolled values are guaranteed to reference earlier rows only.
"""

from __future__ import annotations

import pandas as pd

# Feature names start with "lag_"/"roll_" so the leakage suite can identify them.
LAG_COLUMNS = [1, 7, 14, 21, 28, 35]
ROLL_WINDOWS = [7, 14, 28]
ROLL_FUNCS = ["mean", "std", "max"]


def build_lag_rolling_features(
    df: pd.DataFrame,
    *,
    lags: list[int] | None = None,
    windows: list[int] | None = None,
    funcs: list[str] | None = None,
) -> pd.DataFrame:
    """Add shifted-lag and rolling features to a store-day frame.

    Args:
        df: Store-day rows (all ≤ forecast origin). Sorted by Store, Date.
    Returns:
        A copy with lag/rolling columns appended. NaN where history is short.
    """
    lags = lags if lags is not None else LAG_COLUMNS
    windows = windows if windows is not None else ROLL_WINDOWS
    funcs = funcs if funcs is not None else ROLL_FUNCS

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values(["Store", "Date"]).reset_index(drop=True)

    for lag in lags:
        out[f"lag_{lag}"] = out.groupby("Store")["Sales"].shift(lag)

    for win in windows:
        grp = out.groupby("Store")["Sales"]
        for func in funcs:
            rolled = getattr(grp.shift(1).rolling(win, min_periods=1), func)()
            out[f"roll_{win}_{func}"] = rolled.reset_index(level=0, drop=True)

    return out


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Static + scheduled features that never touch the target.

    All columns here are declared legal by the availability contract
    (static or scheduled classes). ``Customers`` and ``Sales`` are absent.
    """
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out["weekday"] = out["Date"].dt.dayofweek
    out["month"] = out["Date"].dt.month
    out["day_of_year"] = out["Date"].dt.dayofyear
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)
    out["state_holiday_flag"] = (out["StateHoliday"] != "0").astype(int)
    out["store_type_a"] = (out["StoreType"] == "a").astype(int)
    out["store_type_b"] = (out["StoreType"] == "b").astype(int)
    out["store_type_c"] = (out["StoreType"] == "c").astype(int)
    out["store_type_d"] = (out["StoreType"] == "d").astype(int)
    out["assortment_a"] = (out["Assortment"] == "a").astype(int)
    out["assortment_b"] = (out["Assortment"] == "b").astype(int)
    out["assortment_c"] = (out["Assortment"] == "c").astype(int)
    out["promo2_active"] = out["Promo2"].fillna(0).astype(int)
    return out


def build_training_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full training feature set: base + lag/rolling, per-fold safe.

    ``Date`` is kept for joining (models exclude it from feature matrices);
    forbidden columns ``Customers`` and ``Sales`` are dropped from features
    but remain available on the input frame for targets.
    """
    out = build_base_features(df)
    out = build_lag_rolling_features(out)
    keep = [c for c in out.columns if c not in ("Customers", "Sales")]
    return out[keep]
