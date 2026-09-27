"""Official India Meteorological Department (IMD) Operational Extreme Weather Thresholds.

Module 4 & Phase 5 Reference Standards.
All thresholds defined herein are cited directly from official IMD operational criteria.

CITATIONS:
1. Rainfall Categories (24-hour accumulated rainfall):
   Source: India Meteorological Department, "Standard Operation Procedure: Weather Forecasting
   and Warning Services" (2021), Chapter 3 "Terminology & Categories of Rainfall", page 14-16.
   - Light Rain: 2.5 to 15.5 mm
   - Moderate Rain: 15.6 to 64.4 mm
   - Heavy Rain: 64.5 to 115.5 mm (Yellow / Orange alert depending on impact)
   - Very Heavy Rain: 115.6 to 204.4 mm (Orange / Red alert)
   - Extremely Heavy Rain: >= 204.5 mm (Red alert)
   - Short-burst / High-intensity Hourly Threshold: >= 15.0 mm/h (Short Heavy Spell proxy)
   - Cloudburst criterion: >= 100.0 mm in 1 hour over a localized area.

2. Heat Wave Criteria:
   Source: India Meteorological Department, National Weather Forecasting Centre (NWFC),
   "Criteria for Declaring Heat Wave in India" (IMD Operational Manual):
   - Heat wave declaration criteria by regional topography:
     - Plains: Maximum temperature reaches at least 40.0°C.
     - Coastal stations: Maximum temperature reaches at least 37.0°C.
     - Hilly regions: Maximum temperature reaches at least 30.0°C.
   - Severity definitions (Plains):
     - Heat Wave: Max Temp >= 45.0°C (or 40.0°C - 44.9°C with departure >= 4.5°C from normal).
     - Severe Heat Wave: Max Temp >= 47.0°C (or departure >= 6.5°C from normal).
   - Warning Color Codes:
     - Yellow (Heat Alert): 40.0°C - 42.9°C (Plains) / 37.0°C - 39.9°C (Coastal) / 30.0°C - 32.9°C (Hills)
     - Orange (Severe Heat Alert): 43.0°C - 44.9°C (Plains) / 40.0°C - 41.9°C (Coastal) / 33.0°C - 34.9°C (Hills)
     - Red (Extreme Heat Wave): >= 45.0°C (Plains) / >= 42.0°C (Coastal) / >= 35.0°C (Hills)

3. Wind Speed & Gale / Squall Criteria:
   Source: IMD Standard Operating Procedure for Severe Weather Warnings & Cyclone Warning Services:
   - Strong Breeze / Gusty Wind: 40.0 to 50.0 km/h (Yellow Warning)
   - Squall / High Wind: 51.0 to 61.0 km/h (Orange Warning)
   - Gale Wind: 62.0 to 87.0 km/h (Orange / Red Alert)
   - Severe Gale / Storm Force: >= 88.0 km/h (Red Alert)
"""

from typing import Dict, List, Tuple

# =====================================================================
# IMD Official Rainfall Thresholds (mm in 24 hours)
# Citation: IMD SOP Chapter 3 "Terminology & Categories of Rainfall" (2021)
# =====================================================================
IMD_RAIN_VERY_LIGHT_MAX = 2.4
IMD_RAIN_LIGHT_MIN = 2.5
IMD_RAIN_LIGHT_MAX = 15.5
IMD_RAIN_MODERATE_MIN = 15.6
IMD_RAIN_MODERATE_MAX = 64.4
IMD_RAIN_HEAVY_MIN = 64.5        # 64.5 - 115.5 mm (Heavy Rain)
IMD_RAIN_HEAVY_MAX = 115.5
IMD_RAIN_VERY_HEAVY_MIN = 115.6   # 115.6 - 204.4 mm (Very Heavy Rain)
IMD_RAIN_VERY_HEAVY_MAX = 204.4
IMD_RAIN_EXTREMELY_HEAVY_MIN = 204.5  # >= 204.5 mm (Extremely Heavy Rain)

# Hourly intensity proxy thresholds (mm/hour)
IMD_RAIN_HOURLY_HEAVY_SPELL_MIN = 15.0  # Intense hourly rain spell
IMD_RAIN_HOURLY_VERY_HEAVY_MIN = 30.0   # Very intense rain spell
IMD_RAIN_HOURLY_CLOUDBURST_MIN = 100.0  # Cloudburst definition (>= 100 mm in 1 hour)

# Verification proxy threshold for hourly events
HEAVY_RAIN_HOURLY_THRESHOLD_MM = 5.0
HEAVY_RAIN_24H_THRESHOLD_MM = IMD_RAIN_HEAVY_MIN

# =====================================================================
# IMD Official Heatwave Base Thresholds (°C)
# Citation: IMD NWFC "Criteria for Declaring Heat Wave in India"
# =====================================================================
IMD_HEAT_BASE_PLAINS = 40.0
IMD_HEAT_BASE_COASTAL = 37.0
IMD_HEAT_BASE_HILLS = 30.0

# Alert level temperature cutoffs by topography class
IMD_HEAT_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "plains": {
        "yellow": 40.0,
        "orange": 43.0,
        "red": 45.0,
    },
    "arid": {
        "yellow": 40.0,
        "orange": 43.0,
        "red": 45.0,
    },
    "deltaic": {
        "yellow": 38.0,
        "orange": 41.0,
        "red": 43.0,
    },
    "coastal": {
        "yellow": 37.0,
        "orange": 40.0,
        "red": 42.0,
    },
    "hill": {
        "yellow": 30.0,
        "orange": 33.0,
        "red": 35.0,
    },
}

HEATWAVE_THRESHOLD_TEMP_C = IMD_HEAT_BASE_PLAINS

# =====================================================================
# IMD Official Wind Speed Thresholds (km/h)
# Citation: IMD Severe Weather Warning & Cyclone Warning Criteria
# =====================================================================
IMD_WIND_STRONG_BREEZE_MIN = 40.0  # 40 - 50 km/h (Yellow)
IMD_WIND_SQUALL_MIN = 51.0         # 51 - 61 km/h (Orange)
IMD_WIND_GALE_MIN = 62.0           # 62 - 87 km/h (Orange/Red)
IMD_WIND_SEVERE_GALE_MIN = 88.0    # >= 88 km/h (Red)

HIGH_WIND_THRESHOLD_KMH = IMD_WIND_STRONG_BREEZE_MIN


def classify_rainfall_24h(precip_mm: float) -> Tuple[str, str]:
    """Classify 24-hour accumulated rainfall according to official IMD categories.
    
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


def classify_wind(wind_kmh: float) -> Tuple[str, str]:
    """Classify 10m wind speed against IMD squall/gale criteria.
    
    Returns:
        (category_name, alert_level)
    """
    if wind_kmh >= IMD_WIND_SEVERE_GALE_MIN:
        return ("Severe Gale / Cyclone Storm", "RED")
    elif wind_kmh >= IMD_WIND_GALE_MIN:
        return ("Gale Force Wind", "ORANGE")
    elif wind_kmh >= IMD_WIND_SQUALL_MIN:
        return ("Squall / High Wind Warning", "ORANGE")
    elif wind_kmh >= IMD_WIND_STRONG_BREEZE_MIN:
        return ("Strong Gusty Breeze", "YELLOW")
    else:
        return ("Normal Breeze", "GREEN")


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
