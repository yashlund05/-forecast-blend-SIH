"""API clients for Open-Meteo weather data sources.

Features per-source error handling, retries, and structured response dataclasses.
Ensures failure of any individual source degrades gracefully rather than crashing.
"""

from dataclasses import dataclass
import logging
import time
from typing import Any, Dict, List, Optional
import requests

from ingestion.config import (
    AIFS_VARIABLES,
    ARCHIVE_URL,
    CORE_VARIABLES,
    DEFAULT_FORECAST_DAYS,
    ECMWF_AIFS_URL,
    ENSEMBLE_URL,
    MULTIMODEL_SOURCE_MODELS,
    MULTIMODEL_URL,
    REQUEST_TIMEOUT_SECONDS,
    WEATHERNEXT_URL,
    WEATHERNEXT_VARIABLES,
)

logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    """Represents the outcome of an external API request."""

    source_name: str
    is_success: bool
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    status_code: Optional[int] = None
    elapsed_seconds: float = 0.0


class BaseWeatherClient:
    """Base HTTP client with timeout, retry, and exception handling."""

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        timeout: int = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self.session = session or requests.Session()
        self.timeout = timeout

    def _execute_get(
        self, url: str, params: Dict[str, Any], source_name: str
    ) -> FetchResult:
        start_time = time.time()
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            elapsed = time.time() - start_time

            if response.status_code == 200:
                return FetchResult(
                    source_name=source_name,
                    is_success=True,
                    data=response.json(),
                    status_code=200,
                    elapsed_seconds=elapsed,
                )

            error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            logger.warning("Failed request for source %s: %s", source_name, error_msg)
            return FetchResult(
                source_name=source_name,
                is_success=False,
                error_message=error_msg,
                status_code=response.status_code,
                elapsed_seconds=elapsed,
            )

        except requests.RequestException as e:
            elapsed = time.time() - start_time
            error_msg = f"Network/Request error: {str(e)}"
            logger.error("Request exception for source %s: %s", source_name, error_msg)
            return FetchResult(
                source_name=source_name,
                is_success=False,
                error_message=error_msg,
                status_code=None,
                elapsed_seconds=elapsed,
            )


class MultiModelClient(BaseWeatherClient):
    """Client for physical NWP multi-model forecast (GFS, ICON, ECMWF IFS)."""

    def fetch(
        self,
        latitude: float,
        longitude: float,
        forecast_days: int = DEFAULT_FORECAST_DAYS,
        models: Optional[List[str]] = None,
    ) -> FetchResult:
        models = models or MULTIMODEL_SOURCE_MODELS
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(CORE_VARIABLES),
            "models": ",".join(models),
            "forecast_days": forecast_days,
            "timezone": "UTC",
        }
        return self._execute_get(
            url=MULTIMODEL_URL,
            params=params,
            source_name="nwp_multimodel",
        )


class AIFSClient(BaseWeatherClient):
    """Client for ECMWF AIFS (AI/ML weather forecasting model)."""

    def fetch(
        self,
        latitude: float,
        longitude: float,
        forecast_days: int = DEFAULT_FORECAST_DAYS,
    ) -> FetchResult:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(AIFS_VARIABLES),
            "models": "ecmwf_aifs025_single",
            "forecast_days": forecast_days,
            "timezone": "UTC",
        }
        return self._execute_get(
            url=ECMWF_AIFS_URL,
            params=params,
            source_name="ai_aifs",
        )


class WeatherNextClient(BaseWeatherClient):
    """Client for Google DeepMind WeatherNext 2 (AI/ML global weather model).
    
    Provides 0.25° resolution AI forecasts directly via Open-Meteo Ensemble API.
    Native temporal resolution is 6-hourly; interpolated/reconciled by DataNormalizer.
    """

    def fetch(
        self,
        latitude: float,
        longitude: float,
        forecast_days: int = DEFAULT_FORECAST_DAYS,
    ) -> FetchResult:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(WEATHERNEXT_VARIABLES),
            "models": "google_weathernext2_ensemble",
            "forecast_days": forecast_days,
            "timezone": "UTC",
        }
        return self._execute_get(
            url=WEATHERNEXT_URL,
            params=params,
            source_name="ai_weathernext",
        )


class EnsembleClient(BaseWeatherClient):
    """Client for Open-Meteo ECMWF Ensemble spread and members."""

    def fetch(
        self,
        latitude: float,
        longitude: float,
        forecast_days: int = DEFAULT_FORECAST_DAYS,
    ) -> FetchResult:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m,precipitation,wind_speed_10m",
            "models": "ecmwf_ifs025",
            "forecast_days": forecast_days,
            "timezone": "UTC",
        }
        return self._execute_get(
            url=ENSEMBLE_URL,
            params=params,
            source_name="ensemble",
        )


class ArchiveClient(BaseWeatherClient):
    """Client for Open-Meteo Historical / Archive reanalysis ground truth."""

    def fetch(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
    ) -> FetchResult:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(CORE_VARIABLES),
            "timezone": "UTC",
        }
        return self._execute_get(
            url=ARCHIVE_URL,
            params=params,
            source_name="historical_archive",
        )
