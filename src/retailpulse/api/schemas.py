"""API schemas: Pydantic request/response models.

Versioned contract between the FastAPI service and its consumers. Invalid
inputs fail with clear validation messages before touching the model.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    model_version: str
    data_freshness: datetime | None = None


class ForecastResponse(BaseModel):
    store_id: int
    horizon_days: int
    forecasts: list[StoreDayForecast]
    model_version: str
    generated_at: datetime
    note: str = "simulated forecast; see model card for limitations"


class StoreDayForecast(BaseModel):
    date: datetime
    p10: float = Field(ge=0)
    p50: float = Field(ge=0)
    p90: float = Field(ge=0)
    is_open: bool


class StaffingRequest(BaseModel):
    store_id: int = Field(ge=1)
    horizon_days: int = Field(ge=1, le=30, default=30)
    budget_hours: float = Field(ge=0, default=6000.0)
    risk_mode: str = Field(default="p50", pattern="^(p50|p90)$")


class StaffingResponse(BaseModel):
    store_id: int
    risk_mode: str
    budget_hours: float
    allocation: list[StaffingAllocation]
    total_cost: float
    labor_cost: float
    understaff_cost: float
    overstaff_cost: float
    label: str = "simulated cost model"


class StaffingAllocation(BaseModel):
    date: datetime
    scheduled_hours: float = Field(ge=0)


class MonitoringSummary(BaseModel):
    rolling_wape: float | None
    rolling_bias: float | None
    coverage: float | None
    freshness_hours: float | None
    alerts: list[Alert]


class Alert(BaseModel):
    metric: str
    state: str  # warning | failure
    message: str


ForecastResponse.model_rebuild()
StaffingResponse.model_rebuild()
MonitoringSummary.model_rebuild()
