"""Interactive Dashboard for Hybrid AI–NWP Multi-Model Forecast Blending System.

SIH Problem Statement 26081 | Ministry of Earth Sciences / NCMRWF
"""

from datetime import datetime
import json
import os
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add project root to Python module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blending.engine import BLEND_MODELS, ForecastBlendEngine
from ingestion.config import TARGET_LOCATIONS
from ingestion.db import DatabaseManager
from ingestion.pipeline import IngestionPipeline

st.set_page_config(
    page_title="Hybrid AI–NWP Forecast Blend | SIH 26081",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for high credibility and clean layout
st.markdown(
    """
    <style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #6c757d;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #007bff;
        margin-bottom: 10px;
    }
    .status-ok {
        background-color: #d4edda;
        color: #155724;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .status-warn {
        background-color: #fff3cd;
        color: #856404;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .footer {
        margin-top: 50px;
        padding: 20px;
        border-top: 1px solid #e9ecef;
        font-size: 0.85rem;
        color: #6c757d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_db_and_pipeline():
    """Initialize singletons for DatabaseManager and IngestionPipeline."""
    db = DatabaseManager()
    pipeline = IngestionPipeline(db_manager=db)
    blend_engine = ForecastBlendEngine()
    return db, pipeline, blend_engine


db_manager, ingestion_pipeline, blend_engine = get_db_and_pipeline()

# Title and context
st.markdown(
    '<div class="main-header">🌤️ Hybrid AI–NWP Multi-Model Forecast Blending System</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">SIH Problem Statement 26081 | Ministry of Earth Sciences / NCMRWF | Disaster Management</div>',
    unsafe_allow_html=True,
)

# Sidebar controls
st.sidebar.header("🕹️ Forecast Controls")

location_options = {
    cfg.name: loc_id for loc_id, cfg in TARGET_LOCATIONS.items()
}
selected_loc_name = st.sidebar.selectbox(
    "Select Target Location",
    options=list(location_options.keys()),
    index=0,
)
selected_loc_id = location_options[selected_loc_name]
selected_loc_cfg = TARGET_LOCATIONS[selected_loc_id]

st.sidebar.markdown(
    f"""
    **Climate Zone**: `{selected_loc_cfg.zone}`  
    **Topography**: `{selected_loc_cfg.topography}`  
    **Coordinates**: `{selected_loc_cfg.latitude}°N, {selected_loc_cfg.longitude}°E`
    """
)

st.sidebar.markdown("---")
st.sidebar.subheader("📡 Operational Ingestion")

forecast_days = st.sidebar.slider(
    "Forecast Window (Days)",
    min_value=1,
    max_value=7,
    value=7,
    step=1,
)

if st.sidebar.button("⚡ Run Pipeline Now (Selected Location)"):
    with st.spinner(f"Ingesting live forecasts for {selected_loc_name}..."):
        res = ingestion_pipeline.run_live_forecast_ingestion(
            locations=[selected_loc_id],
            forecast_days=forecast_days,
        )
        if res["status"] in ["success", "degraded"]:
            st.sidebar.success(
                f"Ingestion {res['status'].upper()}: {res['sources_succeeded']}/{res['sources_attempted']} sources OK"
            )
        else:
            st.sidebar.error("Ingestion failed. See logs for details.")
        st.rerun()

if st.sidebar.button("🌐 Ingest All 10 Locations"):
    with st.spinner("Ingesting live forecasts across all 10 national locations..."):
        res = ingestion_pipeline.run_live_forecast_ingestion(
            forecast_days=forecast_days,
        )
        st.sidebar.success(
            f"Ingested all locations ({res['sources_succeeded']}/{res['sources_attempted']} sources OK)"
        )
        st.rerun()

# Fetch latest forecast data for selected location
forecasts_df = db_manager.get_latest_forecasts(selected_loc_id)

# System Health & Pipeline Status Banner
pipeline_history = db_manager.get_pipeline_history(limit=1)
col_status1, col_status2, col_status3 = st.columns([2, 2, 3])

with col_status1:
    if not pipeline_history.empty:
        last_run = pipeline_history.iloc[0]
        run_time_utc = last_run["run_timestamp"][:19].replace("T", " ") + " UTC"
        run_status = last_run["status"].upper()
        badge_class = "status-ok" if run_status == "SUCCESS" else "status-warn"
        st.markdown(
            f'Pipeline Status: <span class="{badge_class}">{run_status}</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Last executed: {run_time_utc}")
    else:
        st.markdown(
            'Pipeline Status: <span class="status-warn">AWAITING RUN</span>',
            unsafe_allow_html=True,
        )
        st.caption("No pipeline executions recorded yet.")

with col_status2:
    if not forecasts_df.empty:
        available_models = sorted(forecasts_df["model"].unique().tolist())
        missing_models = [m for m in BLEND_MODELS if m not in available_models]
        st.markdown(f"**Models Available**: `{len(available_models)}/{len(BLEND_MODELS)}`")
        if missing_models:
            st.caption(f"Degraded: missing `{', '.join(missing_models)}`")
        else:
            st.caption("All 4 operational sources active")
    else:
        st.markdown("**Models Available**: `0/4`")
        st.caption("Click 'Run Pipeline Now' to fetch")

with col_status3:
    st.markdown(f"**Target Station**: `{selected_loc_name}` ({selected_loc_cfg.state})")
    st.caption(f"Zone: {selected_loc_cfg.zone} | Lat: {selected_loc_cfg.latitude}, Lon: {selected_loc_cfg.longitude}")

st.markdown("---")

if forecasts_df.empty:
    st.info(
        f"No forecast data is currently stored in SQLite for **{selected_loc_name}**.\n\n"
        "Click the **'Run Pipeline Now'** button in the sidebar to fetch real-time forecasts "
        "from Open-Meteo across GFS, ICON, ECMWF IFS, and ECMWF AIFS."
    )
else:
    # Run dynamic blend
    blend_result = blend_engine.blend(forecasts_df)
    blended_df = blend_result.blended_df

    # Variable Selector Tab
    selected_var_display = st.radio(
        "Forecast Variable",
        options=["Temperature (°C)", "Precipitation (mm)", "Wind Speed (km/h)"],
        horizontal=True,
    )

    var_key_map = {
        "Temperature (°C)": ("temperature_2m", "°C"),
        "Precipitation (mm)": ("precipitation", "mm"),
        "Wind Speed (km/h)": ("wind_speed_10m", "km/h"),
    }
    var_col, unit = var_key_map[selected_var_display]

    # Quick Summary Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        latest_val = blended_df[f"{var_col}_blend"].iloc[0]
        st.metric(
            f"Current {selected_var_display.split()[0]}",
            f"{latest_val:.1f} {unit}",
        )
    with m2:
        max_val = blended_df[f"{var_col}_blend"].max()
        st.metric(
            f"7-Day Peak {selected_var_display.split()[0]}",
            f"{max_val:.1f} {unit}",
        )
    with m3:
        if var_col == "precipitation":
            total_rain = blended_df[f"{var_col}_blend"].sum()
            st.metric("7-Day Cumulative Rain", f"{total_rain:.1f} mm")
        else:
            min_val = blended_df[f"{var_col}_blend"].min()
            st.metric(f"7-Day Min {selected_var_display.split()[0]}", f"{min_val:.1f} {unit}")
    with m4:
        st.metric(
            "Contributing Sources",
            f"{len(blend_result.available_models)} Models",
            delta=f"Missing: {len(blend_result.missing_models)}" if blend_result.missing_models else "Full Blend",
            delta_color="normal" if not blend_result.missing_models else "inverse",
        )

    # Plotly Timeseries Chart
    fig = go.Figure()

    times = pd.to_datetime(blended_df["target_time"])

    # Uncertainty Envelope (Min - Max of individual models)
    fig.add_trace(
        go.Scatter(
            x=times,
            y=blended_df[f"{var_col}_max"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=blended_df[f"{var_col}_min"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(220, 53, 69, 0.12)",
            name="Model Spread Envelope (Min–Max)",
            hoverinfo="skip",
        )
    )

    # Individual Model Traces
    model_styles = {
        "gfs": {"name": "NOAA GFS", "color": "#1f77b4", "dash": "dot"},
        "icon": {"name": "DWD ICON", "color": "#ff7f0e", "dash": "dot"},
        "ecmwf_ifs": {"name": "ECMWF IFS (Physics)", "color": "#2ca02c", "dash": "dash"},
        "ecmwf_aifs": {"name": "ECMWF AIFS (AI/ML)", "color": "#9467bd", "dash": "dashdot"},
    }

    for model_id, style in model_styles.items():
        col_name = f"{var_col}_{model_id}"
        if col_name in blended_df.columns and blended_df[col_name].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=blended_df[col_name],
                    mode="lines",
                    name=style["name"],
                    line=dict(color=style["color"], dash=style["dash"], width=1.8),
                    opacity=0.85,
                )
            )

    # Dynamic Blended Forecast (Thick prominent line)
    fig.add_trace(
        go.Scatter(
            x=times,
            y=blended_df[f"{var_col}_blend"],
            mode="lines+markers",
            marker=dict(size=4),
            name="<b>Dynamic Hybrid Blend</b>",
            line=dict(color="#d62728", width=3.5),
        )
    )

    fig.update_layout(
        title=f"<b>Multi-Model Forecast vs. Dynamic Blend: {selected_loc_name} ({selected_var_display})</b>",
        xaxis_title="Forecast Target Time (UTC)",
        yaxis_title=f"{selected_var_display}",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=520,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Weight Table & Explainability Preview
    col_w, col_data = st.columns([1, 1])

    with col_w:
        st.subheader("⚖️ Active Blend Weights")
        st.caption(
            "Phase 2 baseline uses equal weighting (0.25 each), auto-renormalized "
            "if any source is degraded. In Phase 3, this is replaced by learned "
            "inverse-error and topography-aware weights."
        )
        weights_dict = blend_result.weights_used.get(var_col, {})
        if weights_dict:
            w_df = pd.DataFrame(
                [
                    {
                        "Model": model_styles.get(m, {}).get("name", m),
                        "Canonical ID": m,
                        "Type": "AI / ML" if "aifs" in m else "Physical NWP",
                        "Active Weight": f"{w * 100:.1f}%",
                    }
                    for m, w in sorted(weights_dict.items())
                ]
            )
            st.dataframe(w_df, use_container_width=True, hide_index=True)

    with col_data:
        st.subheader("📊 Forecast Table (Hourly)")
        preview_cols = ["target_time", f"{var_col}_blend"] + [
            f"{var_col}_{m}" for m in BLEND_MODELS if f"{var_col}_{m}" in blended_df.columns
        ]
        st.dataframe(
            blended_df[preview_cols].head(24),
            use_container_width=True,
            hide_index=True,
        )

# Attribution & Compliance Footer (Docs/DATA_SOURCES.md requirement)
st.markdown(
    """
    <div class="footer">
        <b>Data Attribution & Licensing:</b><br>
        Forecast and reanalysis data retrieved via <a href="https://open-meteo.com" target="_blank">Open-Meteo API</a> 
        under <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank">Creative Commons Attribution 4.0 International (CC BY 4.0)</a>.<br>
        Underlying meteorological models:
        <b>ECMWF IFS & AIFS</b> (European Centre for Medium-Range Weather Forecasts),
        <b>NOAA GFS</b> (National Oceanic and Atmospheric Administration),
        <b>DWD ICON</b> (Deutscher Wetterdienst).
        <br><br>
        <i>Notice: Zero synthetic or fabricated skill numbers are displayed. All forecasts reflect real live model ingestions.</i>
    </div>
    """,
    unsafe_allow_html=True,
)
