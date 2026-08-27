"""Stratified subset and champion-selection tests."""

from __future__ import annotations

import pandas as pd

from retailpulse.models.champion import select_champion
from retailpulse.models.subset import stratified_store_subset


def _sample_train(n_stores: int = 20, n_days: int = 400) -> pd.DataFrame:
    rows = []
    for store in range(1, n_stores + 1):
        for i in range(n_days):
            rows.append(
                {
                    "Store": store,
                    "Date": pd.Timestamp("2014-01-01") + pd.Timedelta(days=i),
                    "Sales": float(100 * store + 20 * (i % 7)),
                }
            )
    return pd.DataFrame(rows)


def test_stratified_subset_bounds() -> None:
    train = _sample_train()
    subset = stratified_store_subset(train, max_stores=8)
    assert len(subset) <= 8
    assert len(set(subset)) == len(subset)


def test_stratified_subset_spans_quartiles() -> None:
    train = _sample_train(n_stores=40)
    subset = stratified_store_subset(train, max_stores=12)
    assert len(subset) >= 4  # should span several size/volatility cells


def test_select_champion_prefers_candidate() -> None:
    from retailpulse.evaluation.backtester import BacktestResult, FoldResult

    def make(model: str, wape: float, coverage: float) -> BacktestResult:
        sc = {"wape": wape, "coverage": coverage}
        r = BacktestResult(model=model)
        r.folds.append(
            FoldResult(
                fold_id=0,
                model=model,
                scorecard=sc,
                runtime_seconds=10,
                n_train_rows=10,
                n_valid_rows=10,
            )
        )
        r.aggregate = sc
        return r

    results = [
        make("seasonal_naive", 0.22, 0.77),
        make("lightgbm_quantile", 0.12, 0.82),
        make("tft_challenger", 0.13, 0.80),
    ]
    decision = select_champion(
        results,
        complexity={
            "seasonal_naive": "trivial",
            "lightgbm_quantile": "medium",
            "tft_challenger": "high",
        },
    )
    assert decision.champion == "lightgbm_quantile"
    assert "lightgbm_quantile" in decision.adr_text


def test_select_champion_falls_back_to_baseline() -> None:
    decision = select_champion([], complexity={})
    assert decision.champion == "seasonal_naive"
