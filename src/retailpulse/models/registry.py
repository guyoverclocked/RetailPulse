"""Model registry and promotion gate.

A candidate is promoted only if, on the validation folds:

1. WAPE beats seasonal-naive (``min_skill_vs_naive``),
2. empirical P10-P90 coverage lands inside ``coverage_band``,
3. runtime stays under ``max_backtest_minutes``.

The gate refuses promotion otherwise and records the reason. The sealed
holdout is unlocked ONLY after a champion-selection ADR is written.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from retailpulse.config import get_config
from retailpulse.evaluation.backtester import BacktestResult


@dataclass(frozen=True)
class PromotionDecision:
    """Outcome of the promotion gate for one candidate."""

    model: str
    promoted: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def summary(self) -> str:
        verdict = "PROMOTED" if self.promoted else "REJECTED"
        if self.reasons:
            return f"{self.model}: {verdict} ({'; '.join(self.reasons)})"
        return f"{self.model}: {verdict}"


def evaluate_promotion(
    result: BacktestResult,
    *,
    naive_wape: float,
) -> PromotionDecision:
    """Apply the promotion gate to a backtest result."""
    cfg = get_config().lightgbm.promotion_gate
    reasons: list[str] = []

    if result.aggregate.get("wape") is None:
        return PromotionDecision(result.model, promoted=False, reasons=("no valid folds scored",))

    wape = result.aggregate["wape"]
    skill = 1.0 - wape / naive_wape if naive_wape > 0 else float("inf")
    if skill <= float(cfg["min_skill_vs_naive"]):
        reasons.append(f"skill_vs_naive {skill:.4f} <= gate {cfg['min_skill_vs_naive']}")

    cov = result.aggregate.get("coverage")
    if cov is not None:
        low, high = cfg["coverage_band"]
        if not (low <= cov <= high):
            reasons.append(f"coverage {cov:.4f} outside band [{low}, {high}]")
    else:
        reasons.append("coverage missing")

    total_runtime = sum(f.runtime_seconds for f in result.folds) / 60.0
    if total_runtime > float(cfg["max_backtest_minutes"]):
        reasons.append(f"runtime {total_runtime:.1f}m > gate {cfg['max_backtest_minutes']}m")

    return PromotionDecision(
        model=result.model,
        promoted=not reasons,
        reasons=tuple(reasons),
    )
