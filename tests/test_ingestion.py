"""Unit tests for Ingestion module (Module 1).

Covers:
- Live endpoint validation for physical NWP, AIFS, and Ensemble.
- Schema normalization and canonical model naming.
- Graceful degradation when an individual source fails (Hard Rule 6).
- Database persistence and query retrieval.
"""

from pathlib import Path
import tempfile
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from ingestion.clients import (
    AIFSClient,
    ArchiveClient,
    EnsembleClient,
    FetchResult,
    MultiModelClient,
)
from ingestion.config import TARGET_LOCATIONS
from ingestion.db import DatabaseManager
from ingestion.normalizer import DataNormalizer
from ingestion.pipeline import IngestionPipeline


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for test isolation."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False) as f:
        temp_path = Path(f.name)
    db = DatabaseManager(db_path=temp_path)
    yield db
    try:
        temp_path.unlink()
    except Exception:
        pass


def test_target_locations_config():
    """Verify all 10 required locations are configured with coordinates and topography."""
    assert len(TARGET_LOCATIONS) == 10
    required_locations = [
        "mumbai",
        "delhi",
        "chennai",
        "kolkata",
        "guwahati",
        "jaisalmer",
        "shimla",
        "bhubaneswar",
        "bengaluru",
        "thiruvananthapuram",
    ]
    for loc_id in required_locations:
        assert loc_id in TARGET_LOCATIONS
        cfg = TARGET_LOCATIONS[loc_id]
        assert -90 <= cfg.latitude <= 90
        assert -180 <= cfg.longitude <= 180
        assert cfg.topography in ["coastal", "plains", "arid", "hill/orographic", "deltaic"]


def test_live_multimodel_ingestion_single_location():
    """Verify live MultiModelClient returns valid GFS, ICON, and IFS forecasts."""
    client = MultiModelClient()
    mumbai = TARGET_LOCATIONS["mumbai"]
    res = client.fetch(latitude=mumbai.latitude, longitude=mumbai.longitude, forecast_days=2)

    assert res.is_success is True
    assert res.data is not None
    assert "hourly" in res.data
    hourly = res.data["hourly"]
    assert "temperature_2m_gfs_seamless" in hourly
    assert "temperature_2m_icon_seamless" in hourly
    assert "temperature_2m_ecmwf_ifs025" in hourly


def test_live_aifs_ingestion_single_location():
    """Verify live AIFSClient returns valid ECMWF AIFS data."""
    client = AIFSClient()
    mumbai = TARGET_LOCATIONS["mumbai"]
    res = client.fetch(latitude=mumbai.latitude, longitude=mumbai.longitude, forecast_days=2)

    assert res.is_success is True
    assert res.data is not None
    assert "hourly" in res.data
    hourly = res.data["hourly"]
    assert "temperature_2m" in hourly
    assert "precipitation" in hourly
    assert "wind_speed_10m" in hourly


def test_data_normalizer_multimodel():
    """Verify normalizer maps models to canonical names and standard schema."""
    sample_payload = {
        "hourly": {
            "time": ["2026-09-28T00:00", "2026-09-28T01:00"],
            "temperature_2m_gfs_seamless": [28.5, 29.0],
            "precipitation_gfs_seamless": [0.0, 1.2],
            "wind_speed_10m_gfs_seamless": [12.0, 14.5],
            "wind_gusts_10m_gfs_seamless": [18.0, 22.0],
            "temperature_2m_icon_seamless": [28.0, 28.5],
            "precipitation_icon_seamless": [0.0, 0.5],
            "wind_speed_10m_icon_seamless": [11.0, 13.0],
            "wind_gusts_10m_icon_seamless": [16.0, 19.0],
            "temperature_2m_ecmwf_ifs025": [28.2, 28.8],
            "precipitation_ecmwf_ifs025": [0.0, 0.8],
            "wind_speed_10m_ecmwf_ifs025": [11.5, 13.8],
            "wind_gusts_10m_ecmwf_ifs025": [17.0, 20.5],
        }
    }

    df = DataNormalizer.normalize_multimodel(
        raw_data=sample_payload,
        location_id="mumbai",
        fetch_timestamp="2026-09-27T18:00:00+00:00",
    )

    assert not df.empty
    assert set(df["model"].unique()) == {"gfs", "icon", "ecmwf_ifs"}
    assert set(df["location_id"].unique()) == {"mumbai"}
    assert len(df) == 6  # 2 timestamps * 3 models
    assert "lead_time_hours" in df.columns
    assert df["lead_time_hours"].iloc[0] > 0


def test_ensemble_spread_computation():
    """Verify ensemble normalization computes accurate mean, std, p10, and p90."""
    sample_ensemble = {
        "hourly": {
            "time": ["2026-09-28T00:00"],
            "temperature_2m_member01": [20.0],
            "temperature_2m_member02": [22.0],
            "temperature_2m_member03": [24.0],
            "temperature_2m_member04": [26.0],
            "temperature_2m_member05": [28.0],
        }
    }

    df = DataNormalizer.normalize_ensemble(
        raw_data=sample_ensemble,
        location_id="mumbai",
        fetch_timestamp="2026-09-27T18:00:00+00:00",
    )

    assert not df.empty
    assert len(df) == 1
    row = df.iloc[0]
    assert row["variable"] == "temperature_2m"
    assert row["ensemble_mean"] == pytest.approx(24.0)
    assert row["ensemble_std"] == pytest.approx(np.std([20.0, 22.0, 24.0, 26.0, 28.0]))
    assert row["ensemble_p10"] == pytest.approx(np.percentile([20.0, 22.0, 24.0, 26.0, 28.0], 10))
    assert row["ensemble_p90"] == pytest.approx(np.percentile([20.0, 22.0, 24.0, 26.0, 28.0], 90))


def test_graceful_degradation_on_failed_source(temp_db):
    """Hard Rule 6: If one source fails, pipeline must degrade rather than crash.
    
    Verifies that when AIFS fails (e.g. HTTP 500 / Network down),
    the remaining sources (multimodel, ensemble) still succeed and are stored.
    """
    mock_mm = MagicMock(spec=MultiModelClient)
    mock_mm.fetch.return_value = FetchResult(
        source_name="nwp_multimodel",
        is_success=True,
        data={
            "hourly": {
                "time": ["2026-09-28T00:00"],
                "temperature_2m_gfs_seamless": [30.0],
                "precipitation_gfs_seamless": [0.0],
                "wind_speed_10m_gfs_seamless": [10.0],
                "wind_gusts_10m_gfs_seamless": [15.0],
            }
        },
    )

    # Simulate failing AIFS client
    mock_aifs = MagicMock(spec=AIFSClient)
    mock_aifs.fetch.return_value = FetchResult(
        source_name="ai_aifs",
        is_success=False,
        error_message="HTTP 503: Service Temporarily Unavailable",
        status_code=503,
    )

    mock_ens = MagicMock(spec=EnsembleClient)
    mock_ens.fetch.return_value = FetchResult(
        source_name="ensemble",
        is_success=True,
        data={
            "hourly": {
                "time": ["2026-09-28T00:00"],
                "temperature_2m_member01": [29.5],
                "temperature_2m_member02": [30.5],
            }
        },
    )

    pipeline = IngestionPipeline(
        db_manager=temp_db,
        multimodel_client=mock_mm,
        aifs_client=mock_aifs,
        ensemble_client=mock_ens,
    )

    result = pipeline.run_live_forecast_ingestion(locations=["mumbai"], forecast_days=1)

    # Assert pipeline did not crash and marked status as degraded
    assert result["status"] == "degraded"
    assert result["sources_succeeded"] == 2
    assert result["sources_attempted"] == 3

    # Assert successfully fetched data was stored
    forecasts = temp_db.get_latest_forecasts(location_id="mumbai")
    assert not forecasts.empty
    assert set(forecasts["model"].unique()) == {"gfs"}

    # Assert raw payload cache contains the logged failure
    with temp_db.get_connection() as conn:
        failures = conn.execute(
            "SELECT * FROM raw_payload_cache WHERE is_success = 0"
        ).fetchall()
        assert len(failures) == 1
        assert "503" in failures[0]["error_message"]
