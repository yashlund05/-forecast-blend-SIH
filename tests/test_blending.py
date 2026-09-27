"""Unit tests for Forecast Blending Engine (Module 3).

Covers:
- (a) Weight renormalization when one or more sources are missing/degraded.
- (b) Mathematical bound check: blended value must strictly lie within [min(models), max(models)].
- Pluggable custom weights interface for upcoming Phase 3 learned weights.
"""

import numpy as np
import pandas as pd
import pytest

from blending.engine import (
    BLEND_MODELS,
    DEFAULT_MODEL_WEIGHTS,
    ForecastBlendEngine,
    renormalize_weights,
)
from ingestion.db import DatabaseManager


def test_renormalize_weights_when_source_missing():
    """Requirement 3a: Verify weights renormalize correctly when a source is missing."""
    configured_weights = {
        "ecmwf_ifs": 0.40,
        "ecmwf_aifs": 0.30,
        "gfs": 0.20,
        "icon": 0.10,
    }

    # Case 1: 'icon' and 'ecmwf_aifs' are degraded/missing
    available = ["ecmwf_ifs", "gfs"]
    renorm = renormalize_weights(available, configured_weights)

    # Sum of available weights = 0.40 + 0.20 = 0.60
    assert "icon" not in renorm
    assert "ecmwf_aifs" not in renorm
    assert renorm["ecmwf_ifs"] == pytest.approx(0.40 / 0.60)
    assert renorm["gfs"] == pytest.approx(0.20 / 0.60)
    assert sum(renorm.values()) == pytest.approx(1.0)


def test_renormalize_weights_fallback():
    """Verify fallback to uniform weights if available models have 0 configured weight."""
    available = ["gfs", "icon"]
    configured_weights = {"ecmwf_ifs": 1.0, "ecmwf_aifs": 0.0}
    renorm = renormalize_weights(available, configured_weights)

    assert renorm["gfs"] == pytest.approx(0.5)
    assert renorm["icon"] == pytest.approx(0.5)
    assert sum(renorm.values()) == pytest.approx(1.0)


def test_blend_engine_renormalizes_on_partial_models():
    """Verify BlendEngine applies normalized weights across partial model DataFrames."""
    engine = ForecastBlendEngine()

    # Synthetic forecast with only 2 models present
    sample_df = pd.DataFrame(
        [
            {
                "location_id": "mumbai",
                "fetch_timestamp": "2026-09-28T00:00:00+00:00",
                "target_time": "2026-09-28T01:00:00+00:00",
                "lead_time_hours": 1.0,
                "model": "gfs",
                "temperature_2m": 30.0,
                "precipitation": 2.0,
                "wind_speed_10m": 10.0,
            },
            {
                "location_id": "mumbai",
                "fetch_timestamp": "2026-09-28T00:00:00+00:00",
                "target_time": "2026-09-28T01:00:00+00:00",
                "lead_time_hours": 1.0,
                "model": "icon",
                "temperature_2m": 28.0,
                "precipitation": 4.0,
                "wind_speed_10m": 12.0,
            },
        ]
    )

    result = engine.blend(sample_df)

    assert not result.blended_df.empty
    assert set(result.missing_models) == {"ecmwf_aifs", "ecmwf_ifs"}
    assert set(result.available_models) == {"gfs", "icon"}

    row = result.blended_df.iloc[0]
    # Equal weight 0.5 each
    assert row["temperature_2m_blend"] == pytest.approx(29.0)
    assert row["precipitation_blend"] == pytest.approx(3.0)
    assert row["wind_speed_10m_blend"] == pytest.approx(11.0)


def test_blended_output_within_individual_model_range():
    """Requirement 3b: Blended output must fall strictly within [min(models), max(models)].
    
    Tests against real forecasts stored in the database for Mumbai.
    """
    db = DatabaseManager()
    forecasts = db.get_latest_forecasts("mumbai")

    if forecasts.empty:
        pytest.skip("No forecasts in DB yet; run pipeline first.")

    engine = ForecastBlendEngine()
    result = engine.blend(forecasts)
    df = result.blended_df

    assert not df.empty

    for _, row in df.iterrows():
        for var in ["temperature_2m", "precipitation", "wind_speed_10m"]:
            blend_val = row[f"{var}_blend"]
            min_val = row[f"{var}_min"]
            max_val = row[f"{var}_max"]

            if not np.isnan(blend_val):
                # Blended value must be >= min and <= max within floating point tolerance
                assert min_val - 1e-4 <= blend_val <= max_val + 1e-4, (
                    f"Violation at {row['target_time']} for {var}: "
                    f"blend={blend_val} not in [{min_val}, {max_val}]"
                )

        # Physical constraint: precipitation cannot be negative
        assert row["precipitation_blend"] >= 0.0


def test_pluggable_weights_interface():
    """Verify that custom computed weights dictionary shifts blended values as expected."""
    engine = ForecastBlendEngine()

    sample_df = pd.DataFrame(
        [
            {
                "location_id": "delhi",
                "fetch_timestamp": "2026-09-28T00:00:00+00:00",
                "target_time": "2026-09-28T01:00:00+00:00",
                "lead_time_hours": 1.0,
                "model": "ecmwf_ifs",
                "temperature_2m": 35.0,
                "precipitation": 0.0,
                "wind_speed_10m": 15.0,
            },
            {
                "location_id": "delhi",
                "fetch_timestamp": "2026-09-28T00:00:00+00:00",
                "target_time": "2026-09-28T01:00:00+00:00",
                "lead_time_hours": 1.0,
                "model": "gfs",
                "temperature_2m": 25.0,
                "precipitation": 0.0,
                "wind_speed_10m": 5.0,
            },
        ]
    )

    # Heavily weight ecmwf_ifs (0.9 vs 0.1)
    custom_weights = {"ecmwf_ifs": 0.90, "gfs": 0.10}
    result = engine.blend(sample_df, weights=custom_weights)

    row = result.blended_df.iloc[0]
    expected_temp = 0.9 * 35.0 + 0.1 * 25.0  # 34.0
    assert row["temperature_2m_blend"] == pytest.approx(expected_temp)
