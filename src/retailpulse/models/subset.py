"""Stratified store-subset selection for challenger benchmarks.

Both TFT and Chronos run on the same stratified subset so their comparison
against LightGBM and seasonal-naive is apples-to-apples. Stratification
groups stores by size and volatility quartiles, then samples proportionally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def stratified_store_subset(
    train: pd.DataFrame,
    max_stores: int,
    *,
    seed: int = 42,
) -> list[int]:
    """Pick a stratified store subset.

    Groups by mean-sales quartile and volatility quartile, then samples each
    cell proportionally to its size up to ``max_stores`` total.
    """
    store_stats = (
        train.groupby("Store")["Sales"]
        .agg(["mean", "std"])
        .fillna(0)
        .reset_index()
    )
    store_stats["size_q"] = pd.qcut(
        store_stats["mean"], q=4, labels=False, duplicates="drop"
    )
    store_stats["vol_q"] = pd.qcut(
        store_stats["std"], q=4, labels=False, duplicates="drop"
    )
    store_stats["cell"] = store_stats["size_q"].astype(str) + "_" + store_stats["vol_q"].astype(str)

    rng = np.random.default_rng(seed)
    picked: list[int] = []
    cells = store_stats["cell"].value_counts()
    for cell, n in cells.items():
        cell_stores = store_stats[store_stats["cell"] == cell]["Store"].to_list()
        k = max(1, round(max_stores * n / len(store_stats)))
        k = min(k, len(cell_stores))
        picked.extend(rng.choice(cell_stores, size=k, replace=False).tolist())

    # Trim to exactly max_stores if overshot.
    return sorted(set(picked))[:max_stores]
