"""Classical benchmark: AutoETS and AutoARIMA via StatsForecast.

Local per-store models serve as interpretable benchmarks against the global
LightGBM candidate. Residual-based P10/P90 intervals are added after point
forecasts, and closed stores are forced to deterministic zeros.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from retailpulse.evaluation.backtester import Model

_Q_LOW, _Q_MID, _Q_HIGH = 0.10, 0.50, 0.90


class ClassicalModel(Model):
    """AutoETS (default) or AutoARIMA local model per store."""

    def __init__(self, method: str = "ets", season_length: int = 7) -> None:
        if method not in ("ets", "arima"):
            raise ValueError("method must be 'ets' or 'arima'")
        self.method = method
        self.season_length = season_length
        self.name = f"auto_{method}"

    def fit_predict(
        self,
        train: pd.DataFrame,
        horizon: int,
        valid_dates: list[pd.Timestamp],
        valid_schedule: pd.DataFrame,
    ) -> pd.DataFrame:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA, AutoETS

        train = train.copy()
        train["Date"] = pd.to_datetime(train["Date"])
        train["ds"] = train["Date"]
        train["y"] = train["Sales"]
        train["unique_id"] = train["Store"].astype(str)

        model = AutoETS(season_length=self.season_length) if self.method == "ets" else AutoARIMA(season_length=self.season_length)
        sf = StatsForecast(models=[model], freq="D", n_jobs=-1)
        sf.fit(train[["unique_id", "ds", "y"]])

        pred_raw = sf.predict(h=horizon, level=[int(_Q_LOW * 100), int(_Q_HIGH * 100)])
        if not isinstance(pred_raw, pd.DataFrame):
            pred_raw = pred_raw.to_pandas()
        pred: pd.DataFrame = pred_raw
        # StatsForecast 'level' returns columns like 'AutoETS-lo-90'. Map to ours.
        pred = pred.rename(
            columns={
                model.__class__.__name__: "Sales_q50",
                f"{model.__class__.__name__}-lo-{int(_Q_LOW * 100)}": "Sales_q10",
                f"{model.__class__.__name__}-hi-{int(_Q_HIGH * 100)}": "Sales_q90",
            }
        )
        # Build Store x Date grid and merge forecasts onto it.
        stores = train["unique_id"].unique()
        grid = pd.MultiIndex.from_product([stores, valid_dates], names=["unique_id", "ds"]).to_frame(index=False)
        out = grid.merge(pred, on=["unique_id", "ds"], how="left")
        out["Store"] = out["unique_id"].astype(int)
        out["Date"] = pd.to_datetime(out["ds"])
        out = out[["Store", "Date", "Sales_q10", "Sales_q50", "Sales_q90"]]

        # Deterministic zeros for planned-closed stores.
        sched = valid_schedule.copy()
        sched["Date"] = pd.to_datetime(sched["Date"])
        closed = sched[sched["Open"] == 0][["Store", "Date"]]
        out = out.merge(closed.assign(closed=1), on=["Store", "Date"], how="left")
        closed_mask = out["closed"] == 1
        out.loc[closed_mask, ["Sales_q10", "Sales_q50", "Sales_q90"]] = 0.0
        out = out.drop(columns=["closed"])

        out["Sales_q10"] = np.maximum(out["Sales_q10"].fillna(0), 0)
        out["Sales_q90"] = np.maximum(out["Sales_q90"].fillna(out["Sales_q50"]), out["Sales_q50"])
        out["Sales_q50"] = out["Sales_q50"].fillna(0)
        return out.reset_index(drop=True)
