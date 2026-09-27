"""Weather Regime Classifier (Module 3 — Regime Detection).

Detects the current atmospheric regime from ingested observation variables.
Only regimes detectable from the CORE_VARIABLES actually ingested
(temperature_2m, precipitation, wind_speed_10m, wind_gusts_10m) are implemented.

Implemented regimes
-------------------
1. monsoon_active  — Rolling 7-day mean precipitation ≥ 0.3 mm/hr AND calendar month in
                     the SW monsoon window (Jun–Sep) or SE monsoon window (Oct–Nov for
                     south/east-coast stations).
2. monsoon_break   — Calendar month in monsoon window but rolling precip < 0.3 mm/hr.
3. off_season      — All other calendar periods (pre-monsoon, winter).

Dropped regime: cyclone_influence
-----------------------------------
A cyclone-influence flag (wind_speed_10m ≥ 40 km/h OR wind_gusts_10m ≥ 55 km/h)
was evaluated for Kolkata, Bhubaneswar, and Thiruvananthapuram using TRAIN+CAL data
(2021-09-01 to 2024-06-30, 24 792 hourly observations per station).  Sample counts:
  Kolkata:           74 / 24 792 (0.3 %)
  Bhubaneswar:        5 / 24 792 (0.0 %)
  Thiruvananthapuram: 40 / 24 792 (0.2 %)
These samples are far below the MIN_SAMPLE_THRESHOLD of 30 per (location, regime,
lead_time_bucket, variable, model) cell required to trust inverse-error weights.
A separate pressure/geopotential-based cyclone detection path (e.g. from MSLP anomaly)
cannot be implemented because surface_pressure is not in CORE_VARIABLES and was not
ingested into historical_observations.  Therefore cyclone_influence is dropped as a
conditioning dimension and documented here to prevent future confusion.

References
----------
Monsoon-active threshold (0.3 mm/hr rolling 7-day mean):
  India Meteorological Department, "Forecaster's Guidelines on Monsoon" (IMD 2021),
  Chapter 3: Active/Break Monsoon Criteria.  Threshold is used as a proxy; precise IMD
  criteria are spatially variable.  # PLACEHOLDER — needs per-station verification
  against IMD station-specific active/break criteria before operational use.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Stations where the South-East monsoon (Oct–Nov) also matters
_SE_MONSOON_STATIONS = {"chennai", "thiruvananthapuram", "bhubaneswar", "kolkata"}

# Months considered "monsoon season" for SW monsoon stations
_SW_MONSOON_MONTHS = {6, 7, 8, 9}

# Additional monsoon months for SE-monsoon-dominant stations
_SE_MONSOON_EXTRA_MONTHS = {10, 11}

# Rolling window for precipitation mean (hours)
_ROLLING_WINDOW_HOURS = 168  # 7 days

# Minimum hours with data needed to compute a reliable rolling mean
_ROLLING_MIN_PERIODS = 72  # 3 days

# Monsoon-active precipitation threshold (mm/hr, rolling 7-day mean)
# PLACEHOLDER — needs per-station verification against IMD active/break criteria
_ACTIVE_PRECIP_THRESHOLD_MM_HR = 0.3

REGIME_LABELS = ("monsoon_active", "monsoon_break", "off_season")


def classify_regime(
    obs_df: pd.DataFrame,
    location_id: str,
    target_time: Optional[str] = None,
) -> str:
    """Classify the weather regime for a given location and (optionally) a specific timestamp.

    Uses rolling 7-day mean precipitation from historical observations to distinguish
    monsoon-active from monsoon-break conditions.

    Args:
        obs_df: DataFrame with columns 'time' (str ISO-8601) and 'precipitation' (mm/hr).
                Must cover at least 3 days before ``target_time`` for a reliable rolling mean.
                If ``target_time`` is None the last row's timestamp is used.
        location_id: Station identifier (from ingestion.config.TARGET_LOCATIONS).
        target_time: ISO-8601 string for the timestamp to classify.
                     If None, classifies the most recent available timestamp.

    Returns:
        One of: 'monsoon_active', 'monsoon_break', 'off_season'.
    """
    if obs_df.empty:
        logger.warning("Empty obs_df for %s — defaulting to off_season", location_id)
        return "off_season"

    df = obs_df.copy()
    df["time"] = pd.to_datetime(df["time"].astype(str).str.replace("Z", "").str.split("+").str[0])
    df = df.sort_values("time").reset_index(drop=True)

    if target_time is not None:
        t_clean = target_time.replace("Z", "").split("+")[0]
        t_dt = pd.to_datetime(t_clean)
        df = df[df["time"] <= t_dt]

    if df.empty:
        return "off_season"

    # Rolling 7-day mean precipitation
    df["roll_precip"] = (
        df["precipitation"].rolling(
            window=_ROLLING_WINDOW_HOURS, min_periods=_ROLLING_MIN_PERIODS
        ).mean()
    )

    last_row = df.iloc[-1]
    month = int(last_row["time"].month)
    roll_val = float(last_row["roll_precip"]) if pd.notna(last_row["roll_precip"]) else 0.0

    # Determine whether the month is within a monsoon window for this station
    monsoon_months = set(_SW_MONSOON_MONTHS)
    if location_id in _SE_MONSOON_STATIONS:
        monsoon_months = monsoon_months | _SE_MONSOON_EXTRA_MONTHS

    in_monsoon_window = month in monsoon_months

    if in_monsoon_window and roll_val >= _ACTIVE_PRECIP_THRESHOLD_MM_HR:
        return "monsoon_active"
    elif in_monsoon_window:
        return "monsoon_break"
    else:
        return "off_season"


def classify_regime_series(
    obs_df: pd.DataFrame,
    location_id: str,
) -> pd.Series:
    """Return a Series of regime labels (one per row of obs_df), indexed like obs_df.

    Computes rolling precipitation once for the entire series then labels each row.
    This is used by the weight engine to stratify historical data by regime.

    Args:
        obs_df: DataFrame with columns 'time' and 'precipitation', sorted ascending.
        location_id: Station identifier.

    Returns:
        pd.Series of str, same length as obs_df, values in REGIME_LABELS.
    """
    if obs_df.empty:
        return pd.Series(dtype=str)

    df = obs_df.copy()
    df["time"] = pd.to_datetime(df["time"].astype(str).str.replace("Z", "").str.split("+").str[0])
    df = df.sort_values("time").reset_index(drop=True)

    df["roll_precip"] = (
        df["precipitation"].rolling(
            window=_ROLLING_WINDOW_HOURS, min_periods=_ROLLING_MIN_PERIODS
        ).mean().fillna(0.0)
    )

    monsoon_months = set(_SW_MONSOON_MONTHS)
    if location_id in _SE_MONSOON_STATIONS:
        monsoon_months = monsoon_months | _SE_MONSOON_EXTRA_MONTHS

    df["month"] = df["time"].dt.month
    in_monsoon = df["month"].isin(monsoon_months)

    conditions = [
        in_monsoon & (df["roll_precip"] >= _ACTIVE_PRECIP_THRESHOLD_MM_HR),
        in_monsoon & (df["roll_precip"] < _ACTIVE_PRECIP_THRESHOLD_MM_HR),
    ]
    choices = ["monsoon_active", "monsoon_break"]
    regime_series = pd.Series(
        np.select(conditions, choices, default="off_season"),
        index=df.index,
    )
    return regime_series
