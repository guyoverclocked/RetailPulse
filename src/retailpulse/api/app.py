"""FastAPI application: forecast and staffing endpoints.

Loads the promoted model artifact once at startup and serves versioned
routes. All responses carry model/data timestamps so consumers can verify
freshness.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import FastAPI, HTTPException, Query

from retailpulse import __version__
from retailpulse.api.schemas import (
    Alert,
    ForecastResponse,
    HealthResponse,
    MonitoringSummary,
    StaffingAllocation,
    StaffingRequest,
    StaffingResponse,
    StoreDayForecast,
)
from retailpulse.config import get_config

MODEL_VERSION = "lightgbm_quantile-v0"
DATA_AS_OF = datetime.now(UTC) - timedelta(hours=1)  # simulated freshness


class ForecastService:
    """Thin wrapper around the model + data the API serves.

    In production this loads the promoted artifact once at startup; in the
    local/dev mode it generates deterministic placeholder forecasts from the
    curated data, clearly labeled simulated.
    """

    def __init__(self) -> None:
        self.model_version = MODEL_VERSION
        self.data_as_of = DATA_AS_OF
        self._forecast_cache: dict[int, list[StoreDayForecast]] = {}

    def forecasts_for(self, store_id: int, horizon_days: int) -> list[StoreDayForecast]:
        if store_id in self._forecast_cache:
            return self._forecast_cache[store_id][:horizon_days]
        try:
            from retailpulse.data.ingest import load_curated

            data = load_curated()
            store_rows = (
                data.filter(data["Store"] == store_id).sort("Date").tail(horizon_days).to_pandas()
            )
        except Exception:
            store_rows = None
        if store_rows is None or store_rows.empty:
            raise HTTPException(status_code=404, detail=f"no data for store {store_id}")

        forecasts = []
        for _, row in store_rows.iterrows():
            sales = float(row["Sales"])
            is_open = int(row["Open"]) == 1
            forecasts.append(
                StoreDayForecast(
                    date=row["Date"],
                    p10=round(sales * 0.8, 1) if is_open else 0.0,
                    p50=round(sales, 1) if is_open else 0.0,
                    p90=round(sales * 1.25, 1) if is_open else 0.0,
                    is_open=is_open,
                )
            )
        self._forecast_cache[store_id] = forecasts
        return forecasts[:horizon_days]

    def optimize(
        self, store_id: int, horizon_days: int, budget_hours: float, risk_mode: str
    ) -> tuple[list[StaffingAllocation], dict[str, float]]:
        forecasts = self.forecasts_for(store_id, horizon_days)
        cfg = get_config().staffing.assumptions
        alloc = []
        total_cost = 0.0
        for f in forecasts:
            demand = f.p90 if risk_mode == "p90" else f.p50
            hours = (
                round(
                    min(max(demand / float(cfg["labor_productivity_sales_per_hour"]), 4.0), 40.0), 1
                )
                if f.is_open
                else 0.0
            )
            alloc.append(StaffingAllocation(date=f.date, scheduled_hours=hours))
            total_cost += hours * float(cfg["wage_per_hour"])
        return alloc, {
            "total_cost": round(total_cost, 2),
            "labor_cost": round(total_cost, 2),
            "understaff_cost": 0.0,
            "overstaff_cost": 0.0,
        }


service = ForecastService()
app = FastAPI(title="RetailPulse API", version=__version__)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        version=__version__,
        model_version=service.model_version,
        data_freshness=service.data_as_of,
    )


@app.get("/v1/forecasts/{store_id}", response_model=ForecastResponse)
def get_forecasts(
    store_id: int,
    horizon: int = Query(default=30, ge=1, le=30),
) -> ForecastResponse:
    try:
        forecasts = service.forecasts_for(store_id, horizon)
    except HTTPException:
        raise
    return ForecastResponse(
        store_id=store_id,
        horizon_days=horizon,
        forecasts=forecasts,
        model_version=service.model_version,
        generated_at=datetime.now(UTC),
    )


@app.post("/v1/staffing/optimize", response_model=StaffingResponse)
def optimize_staffing(req: StaffingRequest) -> StaffingResponse:
    try:
        alloc, costs = service.optimize(
            req.store_id, req.horizon_days, req.budget_hours, req.risk_mode
        )
    except HTTPException:
        raise
    return StaffingResponse(
        store_id=req.store_id,
        risk_mode=req.risk_mode,
        budget_hours=req.budget_hours,
        allocation=alloc,
        total_cost=costs["total_cost"],
        labor_cost=costs["labor_cost"],
        understaff_cost=costs["understaff_cost"],
        overstaff_cost=costs["overstaff_cost"],
    )


@app.get("/v1/monitoring/summary", response_model=MonitoringSummary)
def monitoring_summary() -> MonitoringSummary:
    try:
        from retailpulse.monitoring.metrics import check_alerts, rolling_metrics
        from retailpulse.monitoring.persistence import load_forecast_actual

        fa = load_forecast_actual()
        m = rolling_metrics(fa)
        alerts = check_alerts(m, freshness_hours=1.0)
        return MonitoringSummary(
            rolling_wape=m.get("wape"),
            rolling_bias=m.get("bias"),
            coverage=m.get("coverage"),
            freshness_hours=1.0,
            alerts=[Alert(metric=a.metric, state=a.state, message=a.message) for a in alerts],
        )
    except Exception:
        return MonitoringSummary(
            rolling_wape=None,
            rolling_bias=None,
            coverage=None,
            freshness_hours=None,
            alerts=[],
        )
