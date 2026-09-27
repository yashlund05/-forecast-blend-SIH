"""Verified IMD Operational Thresholds for Extreme Weather Evaluation.

Phase 5 Verified IMD Standards.
Re-exports canonical thresholds and classification functions from extremes.thresholds.
"""

from typing import List, Tuple
from extremes.thresholds import (
    HEATWAVE_THRESHOLD_TEMP_C,
    HEAVY_RAIN_24H_THRESHOLD_MM,
    HEAVY_RAIN_HOURLY_THRESHOLD_MM,
    HIGH_WIND_THRESHOLD_KMH,
    IMD_HEAT_THRESHOLDS,
    IMD_RAIN_EXTREMELY_HEAVY_MIN,
    IMD_RAIN_HEAVY_MAX,
    IMD_RAIN_HEAVY_MIN,
    IMD_RAIN_VERY_HEAVY_MAX,
    IMD_RAIN_VERY_HEAVY_MIN,
    IMD_WIND_GALE_MIN,
    IMD_WIND_SEVERE_GALE_MIN,
    IMD_WIND_SQUALL_MIN,
    IMD_WIND_STRONG_BREEZE_MIN,
    classify_heatwave,
    classify_hourly_rainfall,
    classify_rainfall_24h,
    classify_wind,
    resolve_compound_alert,
)

__all__ = [
    "HEAVY_RAIN_HOURLY_THRESHOLD_MM",
    "HEAVY_RAIN_24H_THRESHOLD_MM",
    "HEATWAVE_THRESHOLD_TEMP_C",
    "HIGH_WIND_THRESHOLD_KMH",
    "IMD_RAIN_HEAVY_MIN",
    "IMD_RAIN_HEAVY_MAX",
    "IMD_RAIN_VERY_HEAVY_MIN",
    "IMD_RAIN_VERY_HEAVY_MAX",
    "IMD_RAIN_EXTREMELY_HEAVY_MIN",
    "IMD_HEAT_THRESHOLDS",
    "IMD_WIND_STRONG_BREEZE_MIN",
    "IMD_WIND_SQUALL_MIN",
    "IMD_WIND_GALE_MIN",
    "IMD_WIND_SEVERE_GALE_MIN",
    "classify_rainfall_24h",
    "classify_hourly_rainfall",
    "classify_heatwave",
    "classify_wind",
    "resolve_compound_alert",
]
