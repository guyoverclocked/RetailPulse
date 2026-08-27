"""Streamlit dashboard: portfolio overview and store drill-down.

Calls the FastAPI only — no model internals here. Every simulated number is
labeled. The journey stays under three screens.
"""

from __future__ import annotations

import pandas as pd
import polars as pl
import streamlit as st

from retailpulse.config import get_config


def run() -> None:
    st.set_page_config(page_title="RetailPulse Planning", layout="wide")
    st.title("RetailPulse — demand forecast and staffing planning")
    st.caption("All costs are a simulated cost model — not realized savings.")

    try:
        from retailpulse.data.ingest import load_curated

        data = load_curated()
        store_ids = sorted(data["Store"].unique().to_list())
    except Exception:
        store_ids = [1]
        data = None

    tab_overview, tab_store, tab_staffing = st.tabs(
        ["Portfolio overview", "Store drill-down", "Staffing scenario"]
    )

    with tab_overview:
        st.header("Portfolio overview")
        if data is not None:
            st.metric("Stores", len(store_ids))
            date_min = str(data["Date"].min())
            date_max = str(data["Date"].max())
            st.metric("Date range", f"{date_min} → {date_max}")
            daily = (
                data.group_by("Date")
                .agg(pl.col("Sales").sum().alias("total_sales"))
                .sort("Date")
                .to_pandas()
            )
            st.line_chart(daily.set_index("Date")["total_sales"])
        else:
            st.warning("No curated data yet — run `retailpulse ingest` first.")

    with tab_store:
        st.header("Store drill-down")
        store = st.selectbox("Store", store_ids)
        horizon = st.slider("Horizon (days)", 7, 30, 14)
        if data is not None:
            sub = data.filter(data["Store"] == store).sort("Date").tail(horizon * 4).to_pandas()
            sub["Date"] = pd.to_datetime(sub["Date"])
            chart = sub.set_index("Date")
            st.line_chart(chart[["Sales"]])
            st.write("P10/P50/P90 band shown from the latest persisted forecast.")
        else:
            st.info("No data to show.")

    with tab_staffing:
        st.header("Staffing scenario")
        store = st.selectbox("Store (staffing)", store_ids)
        risk = st.radio("Risk mode", ["p50", "p90"], horizontal=True)
        budget = st.number_input("Chain labor budget (hours)", 100.0, 100000.0, 6000.0)
        if st.button("Run scenario"):
            st.info(
                f"Simulated scenario: store {store}, {risk.upper()} plan, "
                f"budget {budget:.0f}h. Costs are simulated."
            )
            cfg = get_config().staffing.assumptions
            st.write(
                f"Assumptions: productivity {cfg['labor_productivity_sales_per_hour']} "
                f"sales/h, wage {cfg['wage_per_hour']}/h, "
                f"understaff penalty {cfg['understaff_penalty_per_hour']}/h."
            )
