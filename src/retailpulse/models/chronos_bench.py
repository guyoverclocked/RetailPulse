"""Chronos-2 zero-shot benchmark — research ablation.

Runs only if the ``challengers`` extra is installed. Chronos performs
zero-shot forecasting on the same stratified store subset, same folds, same
metrics. It is an ablation, not a production candidate: the report records
setup friction, latency, and calibration alongside accuracy.
"""

from __future__ import annotations

import pandas as pd

from retailpulse.config import get_config
from retailpulse.evaluation.backtester import Model
from retailpulse.models.subset import stratified_store_subset


class ChronosBenchmark(Model):
    """Chronos-2 zero-shot forecasting on a stratified subset."""

    name = "chronos_benchmark"

    def __init__(self, *, max_stores: int | None = None) -> None:
        cfg = get_config().challengers.chronos
        self.max_stores = max_stores or int(cfg["max_stores"])
        self.model_id = str(cfg["model_id"])
        self.prediction_length = int(cfg["prediction_length"])
        self.num_samples = int(cfg["num_samples"])

    def fit_predict(
        self,
        train: pd.DataFrame,
        horizon: int,
        valid_dates: list[pd.Timestamp],
        valid_schedule: pd.DataFrame,
    ) -> pd.DataFrame:
        try:
            import numpy as np
            import torch
            from chronos import ChronosPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Chronos benchmark requires the 'challengers' extra: `uv sync --all-extras`"
            ) from exc

        pipeline = ChronosPipeline.from_pretrained(
            self.model_id, device_map="auto", torch_dtype=torch.bfloat16
        )
        stores = stratified_store_subset(train, self.max_stores)
        sub = train[train["Store"].isin(stores)].copy()
        sub["Date"] = pd.to_datetime(sub["Date"])

        rows: list[pd.DataFrame] = []
        for store in stores:
            s = sub[sub["Store"] == store].sort_values("Date")
            context = torch.tensor(s["Sales"].to_numpy(dtype=np.float32))
            if len(context) < 14:
                continue
            preds = pipeline.predict(
                context=context,
                prediction_length=horizon,
                num_samples=self.num_samples,
            )
            qs = np.quantile(preds.numpy(), [0.1, 0.5, 0.9], axis=0)
            rows.append(
                pd.DataFrame(
                    {
                        "Store": store,
                        "Date": valid_dates[: len(qs[0])],
                        "Sales_q10": qs[0],
                        "Sales_q50": qs[1],
                        "Sales_q90": qs[2],
                    }
                )
            )

        if not rows:
            raise RuntimeError("Chronos: no stores had enough history for the subset")
        return pd.concat(rows, ignore_index=True)
