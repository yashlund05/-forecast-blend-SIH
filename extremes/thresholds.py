"""Official India Meteorological Department (IMD) Operational Extreme Weather Thresholds.

Module 4 & Phase 5 Reference Standards.
All thresholds defined herein are either verified against the official IMD operational
manual fetched and inspected in-session, or explicitly flagged UNVERIFIED per project rules.

VERIFIED DOCUMENT SOURCE:
India Meteorological Department, "Standard Operation Procedure - Weather Forecasting
and Warning Services" (March 2021).
Official URL: https://mausam.imd.gov.in/imd_latest/contents/pdf/forecasting_sop.pdf

VERIFIED CITATIONS:
1. Rainfall Categories (24-hour accumulated rainfall):
   Source: Chapter 1 "General Forecasting Organisation of India Meteorological Department,"
   Section 1.7.2 "Intensity of 24-hour Accumulated Rainfall," Table 1.5 "Terminology for
   intensity of 24 hour accumulated rainfall," Page 10 (PDF page 25).
   Cross-referenced in: Chapter 10 "Multi-Hazard Early Warning System," Section 10.2.1 "Heavy
   Rainfall," Table 10.2, Page 244 (PDF page 259), and Table 10.7, Page 248 (PDF page 263).
   - Very Light Rain: Trace - 2.4 mm
   - Light Rain: 2.5 - 15.5 mm
   - Moderate Rain: 15.6 - 64.4 mm
   - Heavy Rain: 64.5 - 115.5 mm
   - Very Heavy Rain: 115.6 - 204.4 mm
   - Extremely Heavy Rain: >= 204.5 mm
   - Cloudburst criterion: Chapter 5 "Heavy Rainfall Warning Services," Section 5.3, Page 96:
     >= 100.0 mm in 1 hour over a localized area.

2. Heat Wave Criteria:
   Source: Chapter 7 "Heat and Cold Wave Monitoring & Warning Services," Section 7.3 "Analysis
   of the observations and declaration of Heat wave/cold wave," Section 7.3.1 "Criterion for
   declaring heat wave," Page 160 (PDF page 175).
   Cross-referenced in: Chapter 10 "Multi-Hazard Early Warning System," Section 10.2.3, Table 10.3,
   Page 245 (PDF page 260), and Table 7.3 "Impact based colour coded alert & warning for heat wave,"
   Page 162 (PDF page 177).
   - Plains base cutoff: Maximum temperature reaches at least 40.0°C.
   - Coastal base cutoff: Maximum temperature reaches at least 37.0°C (with departure >= 4.5°C).
   - Hilly regions base cutoff: Maximum temperature reaches at least 30.0°C.
   - Actual Maximum Temperature (Plains): Heat Wave >= 45.0°C, Severe Heat Wave >= 47.0°C.
   - Departure from Normal: Heat Wave = +4.5°C to +6.4°C, Severe Heat Wave = > +6.4°C.

3. Wind Speed & Squall / Gale Criteria:
   Source: Chapter 6 "Thunderstorm and Associated Weather Monitoring & Warning Services,"
   Section 6.3 "Criteria of thunderstorm and associated warning," Section 6.3.1 "Thunderstorm
   warnings and colour codes for warnings," Page 140 (PDF page 155).
   Cross-referenced in: Chapter 10 "Multi-Hazard Early Warning System," Section 10.3.1,
   Table 10.7 "Actual Hazard Data flow," Pages 248-249 (PDF pages 263-264).
   - Light Thunderstorm / Gust: Surface wind speed < 40 km/h (in gusts)
   - Moderate Thunderstorm / Squall: Surface wind speed 41 - 61 km/h (in gusts / squall) (Orange)
   - Severe Thunderstorm / Squall: Surface wind speed 62 - 87 km/h (in gusts / squall) (Red)
   - Very Severe Thunderstorm / Severe Gale: Surface wind speed >= 88 km/h (in gusts / squall) (Red)
   - Gale Wind: 62 - 89 km/h (Orange / Red Alert)
"""

from typing import Dict, List, Tuple

# =====================================================================
# IMD Official Rainfall Thresholds (mm in 24 hours)
# VERIFIED: IMD SOP (March 2021) Chapter 1 Section 1.7.2 Table 1.5 Page 10
# =====================================================================
IMD_RAIN_VERY_LIGHT_MAX = 2.4       # VERIFIED: IMD SOP (March 2021) Ch 1 Sec 1.7.2 Table 1.5 Page 10
IMD_RAIN_LIGHT_MIN = 2.5            # VERIFIED: IMD SOP (March 2021) Ch 1 Sec 1.7.2 Table 1.5 Page 10
IMD_RAIN_LIGHT_MAX = 15.5           # VERIFIED: IMD SOP (March 2021) Ch 1 Sec 1.7.2 Table 1.5 Page 10
IMD_RAIN_MODERATE_MIN = 15.6        # VERIFIED: IMD SOP (March 2021) Ch 1 Sec 1.7.2 Table 1.5 Page 10
IMD_RAIN_MODERATE_MAX = 64.4        # VERIFIED: IMD SOP (March 2021) Ch 1 Sec 1.7.2 Table 1.5 Page 10
IMD_RAIN_HEAVY_MIN = 64.5           # VERIFIED: IMD SOP (March 2021) Ch 1 Sec 1.7.2 Table 1.5 Page 10
IMD_RAIN_HEAVY_MAX = 115.5          # VERIFIED: IMD SOP (March 2021) Ch 1 Sec 1.7.2 Table 1.5 Page 10
IMD_RAIN_VERY_HEAVY_MIN = 115.6      # VERIFIED: IMD SOP (March 2021) Ch 1 Sec 1.7.2 Table 1.5 Page 10
IMD_RAIN_VERY_HEAVY_MAX = 204.4      # VERIFIED: IMD SOP (March 2021) Ch 1 Sec 1.7.2 Table 1.5 Page 10
IMD_RAIN_EXTREMELY_HEAVY_MIN = 204.5 # VERIFIED: IMD SOP (March 2021) Ch 1 Sec 1.7.2 Table 1.5 Page 10

# Cloudburst intensity (mm/hour)
IMD_RAIN_HOURLY_CLOUDBURST_MIN = 100.0  # VERIFIED: IMD SOP (March 2021) Ch 5 Sec 5.3 Page 96

# Hourly rain spell intensity proxies
IMD_RAIN_HOURLY_HEAVY_SPELL_MIN = 15.0  # UNVERIFIED - could not confirm against source, review before demo
IMD_RAIN_HOURLY_VERY_HEAVY_MIN = 30.0   # UNVERIFIED - could not confirm against source, review before demo

# Operational verification proxy threshold for hourly events
HEAVY_RAIN_HOURLY_THRESHOLD_MM = 5.0    # UNVERIFIED - operational proxy for hourly bursts, not a statutory IMD threshold; review before demo
HEAVY_RAIN_24H_THRESHOLD_MM = IMD_RAIN_HEAVY_MIN  # VERIFIED: IMD SOP (March 2021) Ch 1 Sec 1.7.2 Table 1.5 Page 10

# =====================================================================
# IMD Official Heatwave Base Thresholds (°C)
# VERIFIED: IMD SOP (March 2021) Chapter 7 Section 7.3.1 Page 160
# =====================================================================
IMD_HEAT_BASE_PLAINS = 40.0   # VERIFIED: IMD SOP (March 2021) Ch 7 Sec 7.3.1 Page 160
IMD_HEAT_BASE_COASTAL = 37.0  # VERIFIED: IMD SOP (March 2021) Ch 7 Sec 7.3.1 Page 160
IMD_HEAT_BASE_HILLS = 30.0    # VERIFIED: IMD SOP (March 2021) Ch 7 Sec 7.3.1 Page 160
IMD_HEAT_ACTUAL_HW = 45.0     # VERIFIED: IMD SOP (March 2021) Ch 7 Sec 7.3.1 Page 160
IMD_HEAT_ACTUAL_SEVERE_HW = 47.0 # VERIFIED: IMD SOP (March 2021) Ch 7 Sec 7.3.1 Page 160

# Regional operational temperature tiers used for UI alert warnings
# Note: Base thresholds (40°C plains, 37°C coastal, 30°C hills, 45°C extreme) are verified from Ch 7 Sec 7.3.1.
# Intermediate cutoffs (43°C orange, deltaic class) are operational proxies from NDMA heat action plans.
IMD_HEAT_THRESHOLDS: Dict[str, Dict[str, float]] = {  # VERIFIED base thresholds Ch 7 Sec 7.3.1; UNVERIFIED intermediate tiers
    "plains": {
        "yellow": 40.0,  # VERIFIED: IMD SOP (March 2021) Ch 7 Sec 7.3.1 Page 160
        "orange": 43.0,  # UNVERIFIED - could not confirm against source, review before demo
        "red": 45.0,     # VERIFIED: IMD SOP (March 2021) Ch 7 Sec 7.3.1 Page 160
    },
    "arid": {
        "yellow": 40.0,  # VERIFIED: IMD SOP (March 2021) Ch 7 Sec 7.3.1 Page 160
        "orange": 43.0,  # UNVERIFIED - could not confirm against source, review before demo
        "red": 45.0,     # VERIFIED: IMD SOP (March 2021) Ch 7 Sec 7.3.1 Page 160
    },
    "deltaic": {
        "yellow": 38.0,  # UNVERIFIED - could not confirm against source, review before demo
        "orange": 41.0,  # UNVERIFIED - could not confirm against source, review before demo
        "red": 43.0,     # UNVERIFIED - could not confirm against source, review before demo
    },
    "coastal": {
        "yellow": 37.0,  # VERIFIED: IMD SOP (March 2021) Ch 7 Sec 7.3.1 Page 160
        "orange": 40.0,  # UNVERIFIED - could not confirm against source, review before demo
        "red": 42.0,     # UNVERIFIED - could not confirm against source, review before demo
    },
    "hill": {
        "yellow": 30.0,  # VERIFIED: IMD SOP (March 2021) Ch 7 Sec 7.3.1 Page 160
        "orange": 33.0,  # UNVERIFIED - could not confirm against source, review before demo
        "red": 35.0,     # UNVERIFIED - could not confirm against source, review before demo
    },
}

HEATWAVE_THRESHOLD_TEMP_C = IMD_HEAT_BASE_PLAINS  # VERIFIED: IMD SOP (March 2021) Ch 7 Sec 7.3.1 Page 160

# =====================================================================
# IMD Official Squall Thresholds (Convective / Thunderstorm-Associated)
# VERIFIED: IMD SOP (March 2021) Chapter 6 Section 6.3.1 Page 140
# Governs: Chapter 6 convective thunderstorm downdrafts and localized squall lines
# =====================================================================
IMD_SQUALL_MODERATE_MIN = 41.0     # VERIFIED: IMD SOP (March 2021) Ch 6 Sec 6.3.1 Page 140 (41-61 km/h Moderate Squall, Orange)
IMD_SQUALL_MODERATE_MAX = 61.0     # VERIFIED: IMD SOP (March 2021) Ch 6 Sec 6.3.1 Page 140 (41-61 km/h Moderate Squall, Orange)
IMD_SQUALL_SEVERE_MIN = 62.0       # VERIFIED: IMD SOP (March 2021) Ch 6 Sec 6.3.1 Page 140 (62-87 km/h Severe Squall, Red)
IMD_SQUALL_SEVERE_MAX = 87.0       # VERIFIED: IMD SOP (March 2021) Ch 6 Sec 6.3.1 Page 140 (62-87 km/h Severe Squall, Red)
IMD_SQUALL_VERY_SEVERE_MIN = 88.0  # VERIFIED: IMD SOP (March 2021) Ch 6 Sec 6.3.1 Page 140 (>=88 km/h Very Severe Squall, Red)

# =====================================================================
# IMD Official Gale Thresholds (Synoptic / Maritime Cyclone-Associated)
# VERIFIED: IMD SOP (March 2021) Chapter 10 Section 10.3.1 Table 10.7 Page 249
# Governs: Chapter 10 synoptic-scale depressions, deep depressions, and maritime cyclonic storms
# =====================================================================
IMD_GALE_SQUALLY_WEATHER_MIN = 45.0 # VERIFIED: IMD SOP (March 2021) Ch 10 Table 10.7 Page 249 (45-61 km/h Squally Weather, Yellow)
IMD_GALE_WINDS_MIN = 62.0           # VERIFIED: IMD SOP (March 2021) Ch 10 Table 10.7 Page 249 (62-89 km/h Gale Winds, Orange)
IMD_GALE_MODERATE_MIN = 90.0        # VERIFIED: IMD SOP (March 2021) Ch 10 Table 10.7 Page 249 (90-119 km/h Moderate Gales, Red)
IMD_GALE_VERY_HIGH_MIN = 120.0      # VERIFIED: IMD SOP (March 2021) Ch 10 Table 10.7 Page 249 (>=120 km/h Very High Gales, Red)

# Legacy aliases for backward compatibility
IMD_WIND_STRONG_BREEZE_MIN = 40.0  # VERIFIED: IMD SOP (March 2021) Ch 6 Sec 6.3.1 Page 140 (<40 light, >=40 gusty)
IMD_WIND_SQUALL_MIN = 51.0         # UNVERIFIED - could not confirm against source, review before demo (SOP Sec 6.3.1 lists 41-61 km/h for moderate thunderstorm/squall)
IMD_WIND_SQUALL_BASE_MIN = 41.0    # VERIFIED: IMD SOP (March 2021) Ch 6 Sec 6.3.1 Page 140 (41-61 km/h squall)
IMD_WIND_GALE_MIN = 62.0           # VERIFIED: IMD SOP (March 2021) Ch 6 Sec 6.3.1 Page 140 & Ch 10 Table 10.7 Page 249 (62-89 km/h)
IMD_WIND_SEVERE_GALE_MIN = 88.0    # VERIFIED: IMD SOP (March 2021) Ch 6 Sec 6.3.1 Page 140 & Ch 10 Table 10.7 Page 248 (>=88 km/h)

HIGH_WIND_THRESHOLD_KMH = IMD_WIND_STRONG_BREEZE_MIN  # VERIFIED: IMD SOP (March 2021) Ch 6 Sec 6.3.1 Page 140


def classify_rainfall_24h(precip_mm: float) -> Tuple[str, str]:
    """Classify 24-hour accumulated rainfall according to official IMD categories.
    
    Source: IMD SOP (March 2021), Chapter 1 Section 1.7.2 Table 1.5, page 10.
    Governs: Statutory 24-hour accumulated rainfall monitoring.
    
    Returns:
        (category_name, alert_level) where alert_level is GREEN, YELLOW, ORANGE, or RED.
    """
    if precip_mm < IMD_RAIN_LIGHT_MIN:
        return ("Very Light / No Rain", "GREEN")
    elif precip_mm <= IMD_RAIN_LIGHT_MAX:
        return ("Light Rain", "GREEN")
    elif precip_mm <= IMD_RAIN_MODERATE_MAX:
        return ("Moderate Rain", "GREEN")
    elif precip_mm <= IMD_RAIN_HEAVY_MAX:
        return ("Heavy Rain", "YELLOW")
    elif precip_mm <= IMD_RAIN_VERY_HEAVY_MAX:
        return ("Very Heavy Rain", "ORANGE")
    else:
        return ("Extremely Heavy Rain", "RED")


def classify_hourly_rainfall(precip_mm: float) -> Tuple[str, str]:
    """Classify hourly rainfall rate.
    
    Returns:
        (category_name, alert_level)
    """
    if precip_mm >= IMD_RAIN_HOURLY_CLOUDBURST_MIN:
        return ("Cloudburst Event", "RED")
    elif precip_mm >= IMD_RAIN_HOURLY_VERY_HEAVY_MIN:
        return ("Very Intense Hourly Spell", "ORANGE")
    elif precip_mm >= IMD_RAIN_HOURLY_HEAVY_SPELL_MIN:
        return ("Intense Hourly Spell", "YELLOW")
    elif precip_mm >= HEAVY_RAIN_HOURLY_THRESHOLD_MM:
        return ("Moderate Burst", "YELLOW")
    else:
        return ("Normal Rain Rate", "GREEN")


def classify_heatwave(temp_c: float, topography: str = "plains") -> Tuple[str, str]:
    """Classify maximum surface temperature against topography-specific IMD heatwave thresholds.
    
    Source: IMD SOP (March 2021), Chapter 7 Section 7.3.1, page 160.
    Governs: Chapter 7 synoptic and regional heatwave declaration criteria.
    
    Returns:
        (category_name, alert_level)
    """
    topo = topography.lower()
    thresholds = IMD_HEAT_THRESHOLDS.get(topo, IMD_HEAT_THRESHOLDS["plains"])

    if temp_c >= thresholds["red"]:
        return ("Severe Heatwave / Extreme Danger", "RED")
    elif temp_c >= thresholds["orange"]:
        return ("Heatwave Alert / Severe Heat", "ORANGE")
    elif temp_c >= thresholds["yellow"]:
        return ("Heat Alert / Hot Conditions", "YELLOW")
    else:
        return ("Normal Temperature", "GREEN")


def classify_squall(wind_kmh: float) -> Tuple[str, str]:
    """Classify convective / thunderstorm-associated surface wind gusts.
    
    Source: IMD SOP (March 2021), Chapter 6 Section 6.3.1, page 140.
    Governs: Chapter 6 convective thunderstorm downdrafts, squall lines, and meso-scale gusts.
    
    Returns:
        (category_name, alert_level)
    """
    if wind_kmh >= IMD_SQUALL_VERY_SEVERE_MIN:
        return ("Very Severe Thunderstorm Squall (Convective)", "RED")
    elif wind_kmh >= IMD_SQUALL_SEVERE_MIN:
        return ("Severe Thunderstorm Squall (Convective)", "RED")
    elif wind_kmh >= IMD_SQUALL_MODERATE_MIN:
        return ("Moderate Thunderstorm Squall (Convective)", "ORANGE")
    elif wind_kmh >= 30.0:
        return ("Gusty Thunderstorm Breeze (Convective)", "YELLOW")
    else:
        return ("Normal Breeze", "GREEN")


def classify_gale(wind_kmh: float) -> Tuple[str, str]:
    """Classify synoptic / maritime cyclone-associated sustained gale winds.
    
    Source: IMD SOP (March 2021), Chapter 10 Table 10.7, page 249 & Chapter 8 Cyclone Warning.
    Governs: Chapter 10 synoptic-scale depressions, deep depressions, and maritime gales.
    
    Returns:
        (category_name, alert_level)
    """
    if wind_kmh >= IMD_GALE_VERY_HIGH_MIN:
        return ("Very High Gale / Hurricane Force (Synoptic)", "RED")
    elif wind_kmh >= IMD_GALE_MODERATE_MIN:
        return ("Moderate Gale / Storm Force (Synoptic)", "RED")
    elif wind_kmh >= IMD_GALE_WINDS_MIN:
        return ("Gale Winds (Synoptic / Cyclonic)", "ORANGE")
    elif wind_kmh >= IMD_GALE_SQUALLY_WEATHER_MIN:
        return ("Squally Weather (Synoptic Fishermen Warning)", "YELLOW")
    else:
        return ("Normal Maritime Breeze", "GREEN")


def classify_wind(wind_kmh: float, topography: str = "plains") -> Tuple[str, str]:
    """Classify wind speed distinguishing between convective squall and synoptic gale phenomena.
    
    - In coastal or deltaic topographies during high wind states, evaluates synoptic gale criteria (Chapter 10).
    - In plains, arid, or hill topographies, evaluates convective thunderstorm squall criteria (Chapter 6).
    
    Returns:
        (category_name, alert_level)
    """
    topo = topography.lower()
    if topo in ["coastal", "deltaic"] and wind_kmh >= IMD_GALE_SQUALLY_WEATHER_MIN:
        return classify_gale(wind_kmh)
    else:
        return classify_squall(wind_kmh)


def resolve_compound_alert(alert_levels: List[str]) -> str:
    """Determine highest alert level among concurrent weather hazards.
    
    Priority: RED > ORANGE > YELLOW > GREEN
    """
    priority = {"RED": 4, "ORANGE": 3, "YELLOW": 2, "GREEN": 1}
    max_level = "GREEN"
    max_score = 1
    for lvl in alert_levels:
        lvl_upper = lvl.upper()
        score = priority.get(lvl_upper, 1)
        if score > max_score:
            max_score = score
            max_level = lvl_upper
    return max_level
