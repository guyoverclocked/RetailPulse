"""Promotion gate unit tests."""

from __future__ import annotations

from retailpulse.evaluation.backtester import BacktestResult, FoldResult
from retailpulse.models.registry import evaluate_promotion


def _result(wape: float, coverage: float | None, runtime_s: float) -> BacktestResult:
    sc: dict[str, float] = {"wape": wape}
    if coverage is not None:
        sc["coverage"] = coverage
    fr = FoldResult(
        fold_id=0,
        model="lightgbm_quantile",
        scorecard=sc,
        runtime_seconds=runtime_s,
        n_train_rows=100,
        n_valid_rows=30,
    )
    r = BacktestResult(model="lightgbm_quantile")
    r.folds.append(fr)
    r.aggregate = sc
    return r


def test_promoted_when_all_gates_pass() -> None:
    r = _result(wape=0.10, coverage=0.80, runtime_s=60)
    d = evaluate_promotion(r, naive_wape=0.20)
    assert d.promoted
    assert not d.reasons


def test_rejected_when_worse_than_naive() -> None:
    r = _result(wape=0.25, coverage=0.80, runtime_s=60)
    d = evaluate_promotion(r, naive_wape=0.20)
    assert not d.promoted
    assert any("skill_vs_naive" in reason for reason in d.reasons)


def test_rejected_when_coverage_out_of_band() -> None:
    r = _result(wape=0.10, coverage=0.60, runtime_s=60)
    d = evaluate_promotion(r, naive_wape=0.20)
    assert not d.promoted
    assert any("coverage" in reason for reason in d.reasons)


def test_rejected_when_runtime_too_long() -> None:
    r = _result(wape=0.10, coverage=0.80, runtime_s=60 * 300)
    d = evaluate_promotion(r, naive_wape=0.20)
    assert not d.promoted
    assert any("runtime" in reason for reason in d.reasons)


def test_rejected_when_no_folds_scored() -> None:
    r = BacktestResult(model="lightgbm_quantile")
    d = evaluate_promotion(r, naive_wape=0.20)
    assert not d.promoted
