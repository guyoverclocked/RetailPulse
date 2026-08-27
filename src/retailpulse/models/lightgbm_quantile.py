"""Global LightGBM quantile candidate.

One model trained across all stores with store identity as a feature.
Separate objectives per quantile (0.10 / 0.50 / 0.90) on the log target, with
leakage-safe features built per fold by the backtester.

Inference is recursive: lag/rolling features for each forecast day are filled
with the previous days' P50 predictions (the standard multi-step direct-plus-
recursive scheme). The model never sees ``Customers`` (observed-late) and only
uses known-future schedule inputs (Open/Promo/holidays) ahead of the origin.
Closed stores are deterministic zeros.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from retailpulse.calibration.quantile import repair_crossings_df
from retailpulse.evaluation.backtester import Model

_Q_LOW, _Q_MID, _Q_HIGH = 0.10, 0.50, 0.90


class LightGBMQuantileModel(Model):
    """Global gradient-boosted quantile forecaster."""

    name = "lightgbm_quantile"

    def __init__(
        self,
        *,
        quantiles: tuple[float, ...] = (_Q_LOW, _Q_MID, _Q_HIGH),
        n_estimators: int = 400,
        learning_rate: float = 0.08,
        num_leaves: int = 63,
        min_child_samples: int = 50,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 50,
        seed: int = 42,
        max_lags: tuple[int, ...] = (1, 7, 14, 21, 28, 35),
        max_roll_windows: tuple[int, ...] = (7, 14, 28),
    ) -> None:
        self.quantiles = quantiles
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = seed
        self.max_lags = max_lags
        self.max_roll_windows = max_roll_windows
        self.models: dict[float, Any] = {}
        self.feature_columns: list[str] = []

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------
    def _derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calendar/static/derived features plus lag/rolling history.

        All columns reference only past-or-present rows: lags are shifted and
        rolling stats use shift(1) windows. Called on the training frame
        (sales real) and recursively on future frames (sales = predictions).
        """
        out = df.copy()
        out["Date"] = pd.to_datetime(out["Date"])

        out["weekday"] = out["Date"].dt.dayofweek
        out["month"] = out["Date"].dt.month
        out["day_of_year"] = out["Date"].dt.dayofyear
        out["is_weekend"] = (out["weekday"] >= 5).astype(int)
        out["state_holiday_flag"] = (out["StateHoliday"] != "0").astype(int)

        if "StoreType" in out.columns:
            for t in ("a", "b", "c", "d"):
                out[f"store_type_{t}"] = (out["StoreType"] == t).astype(int)
        if "Assortment" in out.columns:
            for a in ("a", "b", "c"):
                out[f"assortment_{a}"] = (out["Assortment"] == a).astype(int)
        if "Promo2" in out.columns:
            out["promo2_flag"] = out["Promo2"].fillna(0).astype(int)

        out = out.sort_values(["Store", "Date"]).reset_index(drop=True)
        grp = out.groupby("Store", sort=False)["Sales"]
        for lag in self.max_lags:
            out[f"lag_{lag}"] = grp.shift(lag)
        for win in self.max_roll_windows:
            shifted = grp.shift(1)
            out[f"roll_{win}_mean"] = shifted.rolling(win, min_periods=1).mean().reset_index(level=0, drop=True)
            out[f"roll_{win}_std"] = shifted.rolling(win, min_periods=1).std().reset_index(level=0, drop=True)
        return out

    def _feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Numeric feature columns. Scheduled inputs (Open, Promo,
        SchoolHoliday, holiday flags) are legal known-future features; only
        identity/target/observed-late/raw-string columns are excluded."""
        exclude = {
            "Store",
            "Date",
            "Sales",
            "Customers",
            "StoreType",
            "Assortment",
            "StateHoliday",  # raw string; state_holiday_flag carries the signal
            "DayOfWeek",  # redundant with weekday
            "Promo2",  # raw; promo2_flag carries the signal
        }
        return [c for c in df.columns if c not in exclude]

    # ------------------------------------------------------------------
    # Training + recursive inference
    # ------------------------------------------------------------------
    def _predict_recursive(
        self,
        working: pd.DataFrame,
        future: pd.DataFrame,
        dates: list[pd.Timestamp],
        models: dict[float, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Recursively predict a list of dates.

        ``working`` is the history frame (real sales, ``is_pred`` column
        required); ``future`` is the known-future schedule frame. Returns the
        forecast frame and the extended working frame.
        """
        working = working.copy()
        out_rows: list[pd.DataFrame] = []
        for d in sorted(pd.to_datetime(dates)):
            day = future[future["Date"] == d].copy()
            if day.empty:
                continue
            day["Sales"] = np.nan
            day["is_pred"] = 1
            working = pd.concat([working, day], ignore_index=True)
            working = self._derived_features(working)

            new_rows = working[working["is_pred"] == 1]
            X_new = new_rows[self.feature_columns].to_numpy(dtype=np.float64)

            preds: dict[float, np.ndarray] = {}
            for q in self.quantiles:
                preds[q] = np.expm1(models[q].predict(X_new))

            q10 = preds.get(_Q_LOW, preds[self.quantiles[0]])
            q50 = preds.get(_Q_MID, preds[self.quantiles[1]])
            q90 = preds.get(_Q_HIGH, preds[self.quantiles[-1]])

            out_day = new_rows[["Store", "Date"]].copy()
            out_day["Sales_q10"] = np.maximum(q10, 0)
            out_day["Sales_q50"] = np.maximum(q50, 0)
            out_day["Sales_q90"] = np.maximum(q90, 0)
            closed = new_rows["Open"].to_numpy() == 0
            out_day.loc[closed, ["Sales_q10", "Sales_q50", "Sales_q90"]] = 0.0
            out_rows.append(out_day)

            fill = out_day["Sales_q50"].to_numpy(dtype=np.float64)
            working.loc[working["is_pred"] == 1, "Sales"] = fill

        out = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame(
            columns=["Store", "Date", "Sales_q10", "Sales_q50", "Sales_q90"]
        )
        return out, working

    def fit_predict(
        self,
        train: pd.DataFrame,
        horizon: int,
        valid_dates: list[pd.Timestamp],
        valid_schedule: pd.DataFrame,
    ) -> pd.DataFrame:
        import lightgbm as lgb

        static_cols = [c for c in ("StoreType", "Assortment", "Promo2") if c in train.columns]

        # Training frame: only open store-days; log target for stability.
        feats_train = self._derived_features(train)
        self.feature_columns = self._feature_columns(feats_train)
        open_train = feats_train[feats_train["Open"] == 1]

        # Fit/calibration split by date: the last 60 open days of training
        # serve as the conformal calibration slice; the rest fits the models.
        all_dates = sorted(open_train["Date"].unique())
        calib_cut = all_dates[-60] if len(all_dates) > 120 else all_dates[len(all_dates) // 2]
        fit_part = open_train[open_train["Date"] < calib_cut]
        calib_part = open_train[open_train["Date"] >= calib_cut]

        y_fit = np.log1p(fit_part["Sales"].to_numpy(dtype=np.float64))
        X_fit = fit_part[self.feature_columns].to_numpy(dtype=np.float64)

        models: dict[float, Any] = {}
        for q in self.quantiles:
            models[q] = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                metric="quantile",
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                n_estimators=self.n_estimators,
                seed=self.seed + int(q * 100),
                verbose=-1,
            )
            models[q].fit(X_fit, y_fit)
        self.models = models

        # Conformal corrections learned on the calibration slice.
        from retailpulse.calibration.quantile import conformal_corrections

        corr_lo, corr_hi = 0.0, 0.0
        if not calib_part.empty:
            calib_dates = list(calib_part["Date"].unique())
            hist = feats_train[feats_train["Date"] < calib_cut].copy()
            hist["is_pred"] = 0
            sched_calib = train[train["Date"].isin(calib_dates)][
                ["Store", "Date", "Open", "Promo", "StateHoliday", "SchoolHoliday"]
            ].copy()
            sched_calib["Date"] = pd.to_datetime(sched_calib["Date"])
            if static_cols:
                last_day = train[train["Date"] < calib_cut]
                last_day = last_day[last_day["Date"] == last_day["Date"].max()]
                sched_calib = sched_calib.merge(
                    last_day[["Store", *static_cols]].drop_duplicates("Store"),
                    on="Store",
                    how="left",
                )
            calib_pred, _ = self._predict_recursive(hist, sched_calib, calib_dates, models)
            merged = calib_part[["Store", "Date", "Sales"]].merge(
                calib_pred, on=["Store", "Date"], how="inner"
            )
            corr_lo, corr_hi = conformal_corrections(
                merged["Sales"].to_numpy(dtype=np.float64),
                merged["Sales_q10"].to_numpy(dtype=np.float64),
                merged["Sales_q90"].to_numpy(dtype=np.float64),
            )

        # Future schedule frame: known-future inputs only.
        sched = valid_schedule.copy()
        sched["Date"] = pd.to_datetime(sched["Date"])
        future = sched[["Store", "Date", "Open", "Promo", "StateHoliday", "SchoolHoliday"]].copy()
        if static_cols:
            last_day = train[train["Date"] == train["Date"].max()]
            future = future.merge(
                last_day[["Store", *static_cols]].drop_duplicates("Store"),
                on="Store",
                how="left",
            )

        working = feats_train.copy()
        working["is_pred"] = 0
        out, _ = self._predict_recursive(working, future, valid_dates, models)

        # Apply conformal corrections to open-store forecasts.
        out = out.copy()
        open_mask = out["Sales_q50"] > 0
        out.loc[open_mask, "Sales_q10"] = np.maximum(
            out.loc[open_mask, "Sales_q10"] - corr_lo, 0.0
        )
        out.loc[open_mask, "Sales_q90"] = out.loc[open_mask, "Sales_q90"] + corr_hi
        return repair_crossings_df(out)
