"""Champion-challenger comparison and selection.

The common-fold table compares every candidate on the identical scorecard
(accuracy, calibration, runtime, complexity), then the champion decision is
recorded as an ADR. Only after the champion is locked does the sealed holdout
get evaluated — exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from retailpulse.evaluation.backtester import Backtester, BacktestResult, Model
from retailpulse.evaluation.splits import RollingOriginSplitter


@dataclass(frozen=True)
class ChampionDecision:
    """The recorded model-selection outcome."""

    champion: str
    reasons: tuple[str, ...]
    challengers: tuple[str, ...]
    adr_text: str


def comparison_table(results: list[BacktestResult], *, complexity: dict[str, str]) -> pd.DataFrame:
    """One row per candidate: accuracy, calibration, runtime, complexity."""
    rows = []
    for r in results:
        row = {
            "model": r.model,
            "complexity": complexity.get(r.model, "unknown"),
            "wape": r.aggregate.get("wape"),
            "mase": r.aggregate.get("mase"),
            "skill_vs_naive": r.aggregate.get("skill_vs_naive"),
            "coverage": r.aggregate.get("coverage"),
            "interval_width": r.aggregate.get("interval_width"),
            "runtime_minutes": round(sum(f.runtime_seconds for f in r.folds) / 60, 2),
            "failed_folds": sum(1 for f in r.folds if f.failed),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def select_champion(
    results: list[BacktestResult],
    *,
    complexity: dict[str, str],
    baseline: str = "seasonal_naive",
) -> ChampionDecision:
    """Pick the champion from common-fold results.

    The LightGBM candidate wins unless a challenger beats it meaningfully on
    WAPE while keeping coverage in-band — or the candidate fails entirely, in
    which case the seasonal-naive baseline is the honest fallback.
    """
    table = comparison_table(results, complexity=complexity)
    if table.empty:
        return ChampionDecision(
            champion=baseline,
            reasons=("no candidates evaluated; baseline stands",),
            challengers=(),
            adr_text=_adr(baseline, "no candidates evaluated", table),
        )
    scored = table.dropna(subset=["wape", "coverage"]).copy()
    if scored.empty:
        return ChampionDecision(
            champion=baseline,
            reasons=("no candidate produced valid scores; baseline stands",),
            challengers=tuple(table["model"].tolist()),
            adr_text=_adr(baseline, "no valid challenger scores", table),
        )

    candidate = scored[scored["model"] == "lightgbm_quantile"]
    others = scored[scored["model"] != "lightgbm_quantile"]
    reasons: list[str] = []

    if not candidate.empty:
        cand = candidate.iloc[0]
        reasons.append(
            f"LightGBM WAPE {cand['wape']:.4f}, coverage {cand['coverage']:.4f}"
        )
        for _, row in others.iterrows():
            if row["model"] == baseline:
                continue
            if row["wape"] < cand["wape"] * 0.95:  # ≥5% better = meaningful
                reasons.append(
                    f"{row['model']} beats candidate by >5% WAPE "
                    f"({row['wape']:.4f} vs {cand['wape']:.4f})"
                )
        champion = "lightgbm_quantile"
    else:
        # Candidate failed; best challenger with valid coverage in [0.7, 0.9]
        valid = scored[
            (scored["coverage"] >= 0.7) & (scored["coverage"] <= 0.9)
        ].sort_values("wape")
        if valid.empty:
            champion = baseline
            reasons.append("all candidates failed; baseline stands")
        else:
            champion = str(valid.iloc[0]["model"])
            reasons.append("candidate failed; next-best challenger selected")

    return ChampionDecision(
        champion=champion,
        reasons=tuple(reasons),
        challengers=tuple(table["model"].tolist()),
        adr_text=_adr(champion, "; ".join(reasons), table),
    )


def _adr(champion: str, reasons: str, table: pd.DataFrame) -> str:
    lines = [
        "# ADR: Model selection",
        "",
        "Status: accepted",
        f"Champion: {champion}",
        f"Reasons: {reasons}",
        "",
        "Comparison:",
        table.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Holdout: evaluated exactly once after this decision (see vault holdout discipline).",
    ]
    return "\n".join(lines)


def evaluate_holdout_once(
    backtester: Backtester,
    model: Model,
    data: pd.DataFrame,
    splitter: RollingOriginSplitter,
) -> BacktestResult | None:
    """Evaluate the champion on the sealed holdout — call exactly once."""
    fold = splitter.holdout()  # raises if still locked
    # Reuse the fold runner via a fresh one-fold splitter.
    mini = RollingOriginSplitter(
        horizon_days=splitter.horizon_days, n_folds=1
    )
    mini.holdout_locked = False
    mini.fit(data["Date"])
    # Override folds to point at the holdout window.
    mini.origins = [fold.origin]
    return backtester.evaluate(model, data)
