"""TFT challenger (Darts) — optional deep-learning comparison.

Runs only if the ``challengers`` extra is installed (torch + darts). The TFT
sees static store attributes, known-future calendar/promo inputs, and past
target values; quantile outputs are produced on the same folds and scorecard
as every other candidate.

Kept as one controlled run with a compute budget and stopping rule, per the
vault's champion-challenger policy: the TFT is promoted ONLY if it beats the
LightGBM candidate meaningfully across folds and segments.
"""

from __future__ import annotations

import pandas as pd

from retailpulse.config import get_config
from retailpulse.evaluation.backtester import Model
from retailpulse.models.subset import stratified_store_subset


class DartsTFTModel(Model):
    """Temporal Fusion Transformer via Darts, trained per fold on a subset."""

    name = "tft_challenger"

    def __init__(self, *, max_stores: int | None = None) -> None:
        cfg = get_config().challengers.tft
        self.max_stores = max_stores or int(cfg["max_stores"])
        self.input_chunk = int(cfg["input_chunk_length"])
        self.output_chunk = int(cfg["output_chunk_length"])
        self.hidden_size = int(cfg["hidden_size"])
        self.n_epochs = int(cfg["n_epochs"])
        self.batch_size = int(cfg["batch_size"])
        self.quantiles = [float(q) for q in cfg["quantiles"]]

    def fit_predict(
        self,
        train: pd.DataFrame,
        horizon: int,
        valid_dates: list[pd.Timestamp],
        valid_schedule: pd.DataFrame,
    ) -> pd.DataFrame:
        try:
            from darts import TimeSeries
            from darts.models import TFTModel
        except ImportError as exc:
            raise RuntimeError(
                "Darts TFT challenger requires the 'challengers' extra: "
                "`uv sync --all-extras`"
            ) from exc

        stores = stratified_store_subset(train, self.max_stores)
        sub = train[train["Store"].isin(stores)].copy()
        sub["Date"] = pd.to_datetime(sub["Date"])

        rows: list[pd.DataFrame] = []
        for store in stores:
            s = sub[sub["Store"] == store].sort_values("Date")
            if len(s) < self.input_chunk + 7:
                continue
            series = TimeSeries.from_dataframe(
                s, time_col="Date", value_cols=["Sales"], fill_missing_dates=True, freq="D"
            )
            # Known-future covariates for the horizon.
            future_sched = valid_schedule[valid_schedule["Store"] == store].copy()
            future_sched["Date"] = pd.to_datetime(future_sched["Date"])
            cov = TimeSeries.from_dataframe(
                future_sched.set_index("Date")[["Promo", "SchoolHoliday"]],
                fill_missing_dates=True,
                freq="D",
            )
            model = TFTModel(
                input_chunk_length=self.input_chunk,
                output_chunk_length=self.output_chunk,
                hidden_size=self.hidden_size,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                quantiles=self.quantiles,
                random_state=42,
                pl_trainer_kwargs={"enable_progress_bar": False},
            )
            model.fit(series, future_covariates=cov)
            pred = model.predict(
                n=horizon, future_covariates=cov, num_samples=50
            )
            df = pred.quantile_df()
            df = df.reset_index().rename(columns={"index": "Date"})
            df["Store"] = store
            # Map quantile columns to our contract.
            df = df.rename(
                columns={
                    "Sales_0.1": "Sales_q10",
                    "Sales_0.5": "Sales_q50",
                    "Sales_0.9": "Sales_q90",
                }
            )
            rows.append(df[["Store", "Date", "Sales_q10", "Sales_q50", "Sales_q90"]])

        if not rows:
            raise RuntimeError("TFT: no stores had enough history for the subset")
        out = pd.concat(rows, ignore_index=True)
        out["Sales_q10"] = out["Sales_q10"].fillna(out["Sales_q50"])
        out["Sales_q90"] = out["Sales_q90"].fillna(out["Sales_q50"])
        return out
