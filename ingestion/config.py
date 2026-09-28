"""Configuration for target locations, API endpoints, models, and database paths.

Follows the requirements in docs/DATA_SOURCES.md and docs/PRD.md.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class LocationConfig:
    location_id: str
    name: str
    latitude: float
    longitude: float
    zone: str
    topography: str
    state: str


# 10 Target locations across Indian climate zones as specified in docs/PRD.md and DATA_SOURCES.md
TARGET_LOCATIONS: Dict[str, LocationConfig] = {
    "mumbai": LocationConfig(
        location_id="mumbai",
        name="Mumbai",
        latitude=19.0760,
        longitude=72.8777,
        zone="Coastal",
        topography="coastal",
        state="Maharashtra",
    ),
    "delhi": LocationConfig(
        location_id="delhi",
        name="Delhi",
        latitude=28.6139,
        longitude=77.2090,
        zone="Semi-arid",
        topography="plains",
        state="Delhi",
    ),
    "chennai": LocationConfig(
        location_id="chennai",
        name="Chennai",
        latitude=13.0827,
        longitude=80.2707,
        zone="Coastal/monsoon",
        topography="coastal",
        state="Tamil Nadu",
    ),
    "kolkata": LocationConfig(
        location_id="kolkata",
        name="Kolkata",
        latitude=22.5726,
        longitude=88.3639,
        zone="Deltaic/cyclone-exposed",
        topography="deltaic",
        state="West Bengal",
    ),
    "guwahati": LocationConfig(
        location_id="guwahati",
        name="Guwahati",
        latitude=26.1445,
        longitude=91.7362,
        zone="High-rainfall NE",
        topography="hill/orographic",
        state="Assam",
    ),
    "jaisalmer": LocationConfig(
        location_id="jaisalmer",
        name="Jaisalmer",
        latitude=26.9157,
        longitude=70.9083,
        zone="Arid",
        topography="arid",
        state="Rajasthan",
    ),
    "shimla": LocationConfig(
        location_id="shimla",
        name="Shimla",
        latitude=31.1048,
        longitude=77.1734,
        zone="Hill/orographic",
        topography="hill/orographic",
        state="Himachal Pradesh",
    ),
    "bhubaneswar": LocationConfig(
        location_id="bhubaneswar",
        name="Bhubaneswar",
        latitude=20.2961,
        longitude=85.8245,
        zone="Cyclone-prone east coast",
        topography="coastal",
        state="Odisha",
    ),
    "bengaluru": LocationConfig(
        location_id="bengaluru",
        name="Bengaluru",
        latitude=12.9716,
        longitude=77.5946,
        zone="Plateau",
        topography="plains",
        state="Karnataka",
    ),
    "thiruvananthapuram": LocationConfig(
        location_id="thiruvananthapuram",
        name="Thiruvananthapuram",
        latitude=8.5241,
        longitude=76.9366,
        zone="Coastal monsoon",
        topography="coastal",
        state="Kerala",
    ),
}

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DB_PATH = DATA_DIR / "forecast_blend.sqlite3"

# Open-Meteo Endpoints
MULTIMODEL_URL = "https://api.open-meteo.com/v1/forecast"
ECMWF_AIFS_URL = "https://api.open-meteo.com/v1/ecmwf"
ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHERNEXT_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

# Multi-model forecast parameters
MULTIMODEL_SOURCE_MODELS: List[str] = [
    "gfs_seamless",
    "icon_seamless",
    "ecmwf_ifs025",
]

# Canonical model naming across all ingestion and blending modules
MODEL_MAP = {
    "gfs_seamless": "gfs",
    "icon_seamless": "icon",
    "ecmwf_ifs025": "ecmwf_ifs",
    "ecmwf_aifs025_single": "ecmwf_aifs",
    "google_weathernext2_ensemble": "weathernext",
}

# Weather variables requested across models
CORE_VARIABLES: List[str] = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
]

# AIFS specific variables (wind_gusts_10m is not directly output by AIFS single)
AIFS_VARIABLES: List[str] = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
]

# Google WeatherNext 2 variables
WEATHERNEXT_VARIABLES: List[str] = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
]

# Forecast window settings
DEFAULT_FORECAST_DAYS = 7
REQUEST_TIMEOUT_SECONDS = 15
MAX_RETRIES = 3

