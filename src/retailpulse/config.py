"""Validated configuration loading.

Every YAML file under ``configs/`` is loaded once and validated against Pydantic
models. Modules receive typed config objects; they never read raw dicts. Invalid
config fails hard at import/CLI time with a precise message.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SyntheticConfig(_FrozenModel):
    seed: int
    n_stores: int
    start_date: str
    end_date: str
    store_types: list[str]
    assortments: list[str]
    promo_rate: float
    closure_rate: float
    school_holiday_rate: float
    state_holiday_rate: float
    competition_distance_scale: float
    promo2_rate: float


class FixtureConfig(_FrozenModel):
    n_stores: int
    start_date: str
    end_date: str


class DataConfig(_FrozenModel):
    paths: dict[str, str]
    source: dict[str, str]
    synthetic: SyntheticConfig
    fixture: FixtureConfig


class HoldoutConfig(_FrozenModel):
    locked: bool


class MetricsConfig(_FrozenModel):
    quantiles: list[float]
    coverage_target: float
    segments: list[str]


class BacktestConfig(_FrozenModel):
    horizon_days: int
    n_folds: int
    min_train_days: int
    gap_days: int
    holdout: HoldoutConfig
    metrics: MetricsConfig

    @field_validator("horizon_days", "n_folds", "min_train_days")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be positive")
        return v


class LightGBMConfig(_FrozenModel):
    objective: str
    quantiles: list[float]
    seed: int
    features: dict[str, Any]
    defaults: dict[str, Any]
    tuning: dict[str, Any]
    promotion_gate: dict[str, Any]


class ChallengersConfig(_FrozenModel):
    tft: dict[str, Any]
    chronos: dict[str, Any]


class StaffingConfig(_FrozenModel):
    assumptions: dict[str, float]
    constraints: dict[str, float]
    scenarios: dict[str, str]
    sensitivity: dict[str, list[float]]


class MonitoringConfig(_FrozenModel):
    rolling_window_days: int
    data: dict[str, float]
    model: dict[str, float | list[float]]
    service: dict[str, float]


class Config(BaseModel):
    """Top-level configuration bundle."""

    model_config = ConfigDict(frozen=True)

    data: DataConfig
    backtest: BacktestConfig
    lightgbm: LightGBMConfig = Field(alias="models.lightgbm")
    challengers: ChallengersConfig = Field(alias="models.challengers")
    staffing: StaffingConfig
    monitoring: MonitoringConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file missing: {path}")
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Load and validate all configs once per process."""
    data = _load_yaml(CONFIG_DIR / "data.yaml")
    backtest = _load_yaml(CONFIG_DIR / "backtest.yaml")
    staffing = _load_yaml(CONFIG_DIR / "staffing.yaml")
    monitoring = _load_yaml(CONFIG_DIR / "monitoring.yaml")

    lightgbm = _load_yaml(CONFIG_DIR / "models" / "lightgbm.yaml")
    challengers = _load_yaml(CONFIG_DIR / "models" / "challengers.yaml")

    raw = {
        "data": data,
        "backtest": backtest,
        "staffing": staffing,
        "monitoring": monitoring,
        "models.lightgbm": lightgbm,
        "models.challengers": challengers,
    }
    cfg = Config.model_validate(raw)
    if cfg.lightgbm.quantiles != cfg.backtest.metrics.quantiles:
        raise ValueError("lightgbm.yaml quantiles must match backtest.yaml metrics.quantiles")
    return cfg


def resolve_path(relative: str) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(relative)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def ensure_dir(path: Path) -> Path:
    """Create a directory (idempotent) and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def env_seed() -> int:
    """Fixed seed with a documented escape hatch for experiments."""
    return int(os.environ.get("RETAILPULSE_SEED", get_config().data.synthetic.seed))
