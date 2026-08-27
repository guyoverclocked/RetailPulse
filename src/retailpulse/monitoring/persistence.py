"""Monitoring persistence: forecast-vs-actual tables in DuckDB.

Every forecast is persisted with origin, target date, quantiles, model
version, and run ID. Actuals are joined when available, and rolling metrics
are computed from this single source of truth.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from retailpulse.config import get_config, resolve_path

FORECAST_SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    origin DATE,
    target_date DATE,
    store_id INTEGER,
    model_version VARCHAR,
    run_id VARCHAR,
    q10 DOUBLE,
    q50 DOUBLE,
    q90 DOUBLE,
    PRIMARY KEY (origin, target_date, store_id, model_version, run_id)
)
"""

ACTUAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS actuals (
    store_id INTEGER,
    date DATE,
    sales DOUBLE,
    PRIMARY KEY (store_id, date)
)
"""


def monitoring_db() -> duckdb.DuckDBPyConnection:
    """Open (and initialize) the monitoring DuckDB."""
    cfg = get_config()
    path = resolve_path(cfg.data.paths["duckdb_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(FORECAST_SCHEMA)
    con.execute(ACTUAL_SCHEMA)
    return con


def persist_forecasts(
    forecast: pd.DataFrame,
    *,
    origin: pd.Timestamp,
    model_version: str,
    run_id: str,
) -> None:
    """Write a forecast frame (Store, Date, Sales_q10/50/90) to DuckDB."""
    con = monitoring_db()
    rows = forecast.copy()
    rows["Date"] = pd.to_datetime(rows["Date"])
    rows = rows.rename(
        columns={
            "Date": "target_date",
            "Store": "store_id",
            "Sales_q10": "q10",
            "Sales_q50": "q50",
            "Sales_q90": "q90",
        }
    )
    rows["origin"] = pd.Timestamp(origin).date()
    rows["model_version"] = model_version
    rows["run_id"] = run_id
    con.execute(
        "INSERT OR REPLACE INTO forecasts SELECT origin, target_date, store_id, "
        "model_version, run_id, q10, q50, q90 FROM rows"
    )
    con.close()


def persist_actuals(actuals: pd.DataFrame) -> None:
    """Write actual sales (Store, Date, Sales) to DuckDB."""
    con = monitoring_db()
    rows = actuals.rename(columns={"Store": "store_id", "Date": "date", "Sales": "sales"})
    rows["date"] = pd.to_datetime(rows["date"])
    con.execute("INSERT OR REPLACE INTO actuals SELECT store_id, date, sales FROM rows")
    con.close()


def load_forecast_actual(
    *,
    model_version: str | None = None,
) -> pd.DataFrame:
    """Forecast-vs-actual joined table for rolling metrics."""
    con = monitoring_db()
    q = """
        SELECT f.*, a.sales AS actual
        FROM forecasts f
        LEFT JOIN actuals a
          ON f.store_id = a.store_id AND f.target_date = a.date
        WHERE f.model_version = COALESCE(?, f.model_version)
    """
    df = con.execute(q, [model_version]).fetch_df()
    con.close()
    return df
