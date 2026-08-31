"""Rolling-origin backtest splits.

Expanding-window folds that recreate real forecasting decisions: each fold
trains on everything up to an origin, then validates the next ``horizon_days``.
The final 30 days are sealed as a holdout and only unlocked after the
champion-selection ADR (``holdout.locked`` in config).

Every model receives identical folds. A unit test proves no training row's
timestamp crosses its fold's origin.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from retailpulse.config import get_config


class HoldoutLockedError(RuntimeError):
    """Raised when anything tries to touch the sealed holdout before unlock."""


@dataclass(frozen=True)
class Fold:
    """One rolling-origin fold."""

    fold_id: int
    origin: pd.Timestamp  # last training date (exclusive boundary)
    train_dates: tuple[pd.Timestamp, pd.Timestamp]  # inclusive range
    valid_dates: tuple[pd.Timestamp, pd.Timestamp]  # inclusive range
    is_holdout: bool = False


class RollingOriginSplitter:
    """Expanding-window splitter with a sealed final holdout."""

    def __init__(
        self,
        *,
        horizon_days: int | None = None,
        n_folds: int | None = None,
        min_train_days: int | None = None,
        gap_days: int | None = None,
    ) -> None:
        cfg = get_config().backtest
        self.horizon_days = cfg.horizon_days if horizon_days is None else horizon_days
        self.n_folds = cfg.n_folds if n_folds is None else n_folds
        self.min_train_days = cfg.min_train_days if min_train_days is None else min_train_days
        self.gap_days = cfg.gap_days if gap_days is None else gap_days
        self.holdout_locked = cfg.holdout.locked

        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        if self.n_folds <= 0:
            raise ValueError("n_folds must be positive")
        if self.min_train_days <= 0:
            raise ValueError("min_train_days must be positive")
        if self.gap_days < 0:
            raise ValueError("gap_days must be non-negative")

    def fit(self, dates: pd.Series) -> None:
        """Compute fold boundaries from the full date range."""
        self.dates = (
            pd.Series(pd.to_datetime(dates)).sort_values().drop_duplicates().reset_index(drop=True)
        )
        if self.dates.empty:
            raise ValueError("cannot create folds from an empty date series")
        self.date_min = self.dates.min()
        self.date_max = self.dates.max()

        # Always define the holdout boundary. It is useful for callers that
        # explicitly unlock the holdout after fitting the splitter.
        self.holdout_start = self.date_max - pd.Timedelta(days=self.horizon_days - 1)
        if self.holdout_locked:
            # Reserve the last horizon_days as the sealed holdout.
            usable_end = self.holdout_start - pd.Timedelta(days=1)
        else:
            usable_end = self.date_max

        # Fold origins are spaced by one validation window plus the configured
        # gap. The last validation window ends on usable_end, so a locked
        # holdout can never be included in a training fold's validation rows.
        earliest_origin = self.date_min + pd.Timedelta(days=self.min_train_days - 1)
        block = self.horizon_days + self.gap_days
        self.origins: list[pd.Timestamp] = []
        for i in range(self.n_folds):
            origin = usable_end - pd.Timedelta(days=block * (self.n_folds - i))
            if origin < earliest_origin:
                continue
            self.origins.append(origin)
        self.n_effective_folds = len(self.origins)

        if not self.origins:
            raise ValueError(
                f"no fold fits: need >= {self.min_train_days} training days plus "
                f"{self.n_folds}x({self.horizon_days}+{self.gap_days}) days, "
                f"have {(self.date_max - self.date_min).days + 1}"
            )

    def folds(self) -> list[Fold]:
        """Return the training folds (never the holdout)."""
        out = []
        for i, origin in enumerate(self.origins):
            valid_start = origin + pd.Timedelta(days=self.gap_days + 1)
            valid_end = valid_start + pd.Timedelta(days=self.horizon_days - 1)
            out.append(
                Fold(
                    fold_id=i,
                    origin=origin,
                    train_dates=(self.date_min, origin),
                    valid_dates=(valid_start, valid_end),
                )
            )
        return out

    def holdout(self) -> Fold:
        """The sealed final window (requires unlocked state)."""
        if self.holdout_locked:
            raise HoldoutLockedError(
                "holdout is sealed; unlock only after the champion-selection ADR"
            )
        return Fold(
            fold_id=-1,
            origin=self.holdout_start - pd.Timedelta(days=1),
            train_dates=(self.date_min, self.holdout_start - pd.Timedelta(days=1)),
            valid_dates=(self.holdout_start, self.date_max),
            is_holdout=True,
        )

    def unlock_holdout(self) -> None:
        """Flip the holdout open. Call ONLY from the champion-selection step."""
        cfg = get_config().backtest
        if not cfg.holdout.locked:
            return  # already unlocked at config level
        self.holdout_locked = False

    def train_mask(self, df: pd.DataFrame, fold: Fold) -> pd.Series:
        """Boolean mask of rows in the fold's training window."""
        return (df["Date"] >= fold.train_dates[0]) & (df["Date"] <= fold.train_dates[1])

    def valid_mask(self, df: pd.DataFrame, fold: Fold) -> pd.Series:
        """Boolean mask of rows in the fold's validation window."""
        return (df["Date"] >= fold.valid_dates[0]) & (df["Date"] <= fold.valid_dates[1])


def assert_no_leakage(train: pd.DataFrame, fold: Fold) -> None:
    """Hard check: no training row may be dated after the fold origin."""
    bad = train[train["Date"] > fold.origin]
    if not bad.empty:
        raise ValueError(
            f"fold {fold.fold_id}: {len(bad)} training rows dated after origin {fold.origin}"
        )
