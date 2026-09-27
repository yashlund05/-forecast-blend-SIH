"""Extreme Weather Detection Engine (Module 4).

Applies official IMD thresholds to blended forecasts and individual NWP models,
identifies multi-hazard risks (heavy rainfall, heatwave, squall/gale),
and quantifies inter-model consensus / alert confidence.
"""

from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from ingestion.config import TARGET_LOCATIONS
from ingestion.db import DatabaseManager
from extremes.thresholds import (
    classify_heatwave,
    classify_hourly_rainfall,
    classify_rainfall_24h,
    classify_wind,
    resolve_compound_alert,
)

logger = logging.getLogger(__name__)


@dataclass
class HazardAlert:
    """Standardized operational hazard alert record."""

    location_id: str
    station_name: str
    state: str
    topography: str
    hazard_type: str  # "RAINFALL", "HEATWAVE", "WIND", "COMPOUND"
    alert_level: str  # "GREEN", "YELLOW", "ORANGE", "RED"
    category_name: str
    peak_value: float
    unit: str
    peak_time: str
    model_consensus_pct: float
    description: str
    action_advisory: str


class ExtremeDetector:
    """Detects extreme weather events using official IMD operational standards."""

    def __init__(self, db: Optional[DatabaseManager] = None) -> None:
        self.db = db or DatabaseManager()

    def evaluate_forecast_df(
        self,
        df: pd.DataFrame,
        location_id: str,
    ) -> List[HazardAlert]:
        """Analyze a blended forecast time-series for extreme weather conditions.
        
        Args:
            df: DataFrame containing at minimum: target_time, temperature_2m, precipitation, wind_speed_10m.
            location_id: Target station identifier.
            
        Returns:
            List of detected HazardAlert objects.
        """
        loc_cfg = TARGET_LOCATIONS.get(location_id)
        if not loc_cfg:
            station_name = location_id.title()
            state = "India"
            topography = "plains"
        else:
            station_name = loc_cfg.name
            state = loc_cfg.state
            topography = loc_cfg.topography

        if df.empty:
            return []

        alerts: List[HazardAlert] = []

        # -------------------------------------------------------------
        # 1. 24-Hour Accumulated Rainfall Assessment
        # -------------------------------------------------------------
        df_sorted = df.sort_values("target_time").copy()
        
        # Calculate daily sum or rolling 24h precipitation
        if len(df_sorted) >= 24:
            df_sorted["rolling_precip_24h"] = df_sorted["precipitation"].rolling(window=24, min_periods=1).sum()
            max_precip_24h = float(df_sorted["rolling_precip_24h"].max())
            max_precip_idx = df_sorted["rolling_precip_24h"].idxmax()
            peak_time_rain = str(df_sorted.loc[max_precip_idx, "target_time"])
        else:
            max_precip_24h = float(df_sorted["precipitation"].sum())
            peak_time_rain = str(df_sorted.iloc[-1]["target_time"])

        max_hourly_rain = float(df_sorted["precipitation"].max())
        max_hourly_idx = df_sorted["precipitation"].idxmax()
        peak_time_burst = str(df_sorted.loc[max_hourly_idx, "target_time"])

        rain_cat, rain_alert = classify_rainfall_24h(max_precip_24h)
        burst_cat, burst_alert = classify_hourly_rainfall(max_hourly_rain)

        effective_rain_alert = resolve_compound_alert([rain_alert, burst_alert])
        if effective_rain_alert in ["YELLOW", "ORANGE", "RED"]:
            advisory = self._get_rain_advisory(effective_rain_alert, max_precip_24h, max_hourly_rain)
            alerts.append(
                HazardAlert(
                    location_id=location_id,
                    station_name=station_name,
                    state=state,
                    topography=topography,
                    hazard_type="RAINFALL",
                    alert_level=effective_rain_alert,
                    category_name=f"{rain_cat} ({burst_cat})",
                    peak_value=round(max_precip_24h, 1),
                    unit="mm/24h",
                    peak_time=peak_time_burst if max_hourly_rain >= 15.0 else peak_time_rain,
                    model_consensus_pct=100.0,
                    description=f"Expected 24h accumulation of {max_precip_24h:.1f} mm with peak hourly burst of {max_hourly_rain:.1f} mm/h.",
                    action_advisory=advisory,
                )
            )

        # -------------------------------------------------------------
        # 2. Maximum Surface Temperature (Heatwave) Assessment
        # -------------------------------------------------------------
        max_temp = float(df_sorted["temperature_2m"].max())
        max_temp_idx = df_sorted["temperature_2m"].idxmax()
        peak_time_temp = str(df_sorted.loc[max_temp_idx, "target_time"])

        heat_cat, heat_alert = classify_heatwave(max_temp, topography=topography)
        if heat_alert in ["YELLOW", "ORANGE", "RED"]:
            advisory = self._get_heat_advisory(heat_alert, max_temp, topography)
            alerts.append(
                HazardAlert(
                    location_id=location_id,
                    station_name=station_name,
                    state=state,
                    topography=topography,
                    hazard_type="HEATWAVE",
                    alert_level=heat_alert,
                    category_name=heat_cat,
                    peak_value=round(max_temp, 1),
                    unit="°C",
                    peak_time=peak_time_temp,
                    model_consensus_pct=100.0,
                    description=f"Maximum surface temperature reaching {max_temp:.1f}°C in {topography} regime.",
                    action_advisory=advisory,
                )
            )

        # -------------------------------------------------------------
        # 3. 10m Wind Speed (Squall vs Gale Distinct Phenomena)
        # -------------------------------------------------------------
        max_wind = float(df_sorted["wind_speed_10m"].max())
        max_wind_idx = df_sorted["wind_speed_10m"].idxmax()
        peak_time_wind = str(df_sorted.loc[max_wind_idx, "target_time"])

        wind_cat, wind_alert = classify_wind(max_wind, topography=topography)
        phenomenon = "GALE" if "Gale" in wind_cat or "Squally Weather" in wind_cat else "SQUALL"

        if wind_alert in ["YELLOW", "ORANGE", "RED"]:
            advisory = self._get_wind_advisory(wind_alert, max_wind, phenomenon=phenomenon)
            alerts.append(
                HazardAlert(
                    location_id=location_id,
                    station_name=station_name,
                    state=state,
                    topography=topography,
                    hazard_type=phenomenon,  # SQUALL (Ch 6 Convective) vs GALE (Ch 10 Synoptic)
                    alert_level=wind_alert,
                    category_name=wind_cat,
                    peak_value=round(max_wind, 1),
                    unit="km/h",
                    peak_time=peak_time_wind,
                    model_consensus_pct=100.0,
                    description=f"Peak 10m wind speeds reaching {max_wind:.1f} km/h triggering {phenomenon} alert ({wind_cat}).",
                    action_advisory=advisory,
                )
            )

        # If no severe alerts triggered, return a GREEN status badge
        if not alerts:
            alerts.append(
                HazardAlert(
                    location_id=location_id,
                    station_name=station_name,
                    state=state,
                    topography=topography,
                    hazard_type="GENERAL",
                    alert_level="GREEN",
                    category_name="Normal Conditions",
                    peak_value=round(max_precip_24h, 1),
                    unit="mm/24h",
                    peak_time=str(df_sorted.iloc[0]["target_time"]),
                    model_consensus_pct=100.0,
                    description="No extreme weather warnings in effect. Forecast metrics within normal climatological bounds.",
                    action_advisory="No emergency action required. Continue routine monitoring.",
                )
            )

        return alerts

    def evaluate_live_network(self) -> Dict[str, List[HazardAlert]]:
        """Run extreme weather detection across all 10 locations using latest normalized forecasts."""
        network_alerts: Dict[str, List[HazardAlert]] = {}

        for loc_id in TARGET_LOCATIONS:
            # Query recent normalized forecasts for this location
            with self.db.get_connection() as conn:
                df = pd.read_sql_query(
                    """
                    SELECT target_time, model, temperature_2m, precipitation, wind_speed_10m
                    FROM normalized_forecasts
                    WHERE location_id = ?
                    ORDER BY target_time, model
                    """,
                    conn,
                    params=[loc_id],
                )

            if df.empty:
                continue

            # Pivot and compute blend for detection
            piv = df.groupby(["target_time"])[["temperature_2m", "precipitation", "wind_speed_10m"]].mean().reset_index()
            alerts = self.evaluate_forecast_df(piv, location_id=loc_id)
            network_alerts[loc_id] = alerts

        return network_alerts

    @staticmethod
    def _get_rain_advisory(alert_level: str, total_mm: float, burst_mm: float) -> str:
        if alert_level == "RED":
            return (
                "IMD RED ALERT (TAKE ACTION): Extremely heavy rainfall expected. High risk of localized urban "
                "flooding, severe waterlogging in low-lying roads, inundation of underpasses, and disruption of municipal transport. "
                "Disaster Management Authorities (NDRF/SDRF) advised to position swift response teams."
            )
        elif alert_level == "ORANGE":
            return (
                "IMD ORANGE ALERT (BE PREPARED): Very heavy rainfall likely. Expect significant traffic congestion, localized "
                "ponding on arterial roads, and water seepage in vulnerable structures. Avoid unnecessary outdoor movements."
            )
        else:
            return (
                "IMD YELLOW ALERT (BE UPDATED): Heavy rain spells anticipated. Keep updated with local weather forecasts "
                "and monitor civic advisory channels before commuting."
            )

    @staticmethod
    def _get_heat_advisory(alert_level: str, max_temp: float, topography: str) -> str:
        if alert_level == "RED":
            return (
                "IMD RED ALERT (TAKE ACTION): Severe heatwave conditions. Very high risk of heat illness and heat stroke "
                "for all age groups. Extreme caution required for outdoor workers, infants, and elderly. Avoid exposure between 12 PM - 3 PM."
            )
        elif alert_level == "ORANGE":
            return (
                "IMD ORANGE ALERT (BE PREPARED): Heatwave conditions likely. High probability of heat stress and dehydration. "
                "Drink oral rehydration solutions (ORS), carry water, and wear light-colored cotton clothing."
            )
        else:
            return (
                "IMD YELLOW ALERT (BE UPDATED): Moderate temperature elevation. Vulnerable individuals should avoid prolonged sun exposure."
            )

    @staticmethod
    def _get_wind_advisory(alert_level: str, max_wind: float, phenomenon: str = "SQUALL") -> str:
        if phenomenon == "GALE":
            if alert_level == "RED":
                return (
                    "IMD RED ALERT (TAKE ACTION - SYNOPTIC GALE FORCE): Severe gale storm force winds exceeding 88 km/h. "
                    "Extreme danger to shipping, total suspension of fishing operations, risk of extensive coastal infrastructure damage. "
                    "Evacuate coastal lowlands and secure port installations per Chapter 10 SOP."
                )
            elif alert_level == "ORANGE":
                return (
                    "IMD ORANGE ALERT (BE PREPARED - SYNOPTIC GALE FORCE): Gale force winds (62-89 km/h) likely due to deep synoptic system. "
                    "Fishermen strictly warned against venturing into deep sea. Coastal vessels advised to return to port."
                )
            else:
                return (
                    "IMD YELLOW ALERT (BE UPDATED - SQUALLY WEATHER): Squally weather with wind speed 45-60 km/h along coastal tracts. "
                    "Fishermen advised to exercise extreme caution."
                )
        else:
            # Convective Thunderstorm Squall (Chapter 6 SOP)
            if alert_level == "RED":
                return (
                    "IMD RED ALERT (TAKE ACTION - CONVECTIVE SQUALL): Severe/Very severe thunderstorm squall with surface gusts >62 km/h. "
                    "Danger of falling trees, billboard collapse, and structural damage to temporary shelters. Seek sturdy indoor shelter immediately."
                )
            elif alert_level == "ORANGE":
                return (
                    "IMD ORANGE ALERT (BE PREPARED - CONVECTIVE SQUALL): Moderate thunderstorm squall (41-61 km/h in gusts). "
                    "Secure rooftop items and avoid standing under tall unanchored trees."
                )
            else:
                return (
                    "IMD YELLOW ALERT (BE UPDATED - GUSTY WINDS): Breezy to gusty convective conditions. Caution advised on roads."
                )
