"""Unit tests for Skill and Weight Engine (Module 2).

Covers:
- (a) Sum-to-one constraint: weights sum to 1.0 per (location, season, lead_time, variable) bucket.
- (b) Low-sample fallback: buckets with fewer than MIN_SAMPLE_THRESHOLD fall back to equal weighting and flag low_confidence.
- (c) Anti-leakage enforcement: Hard Rule 4 assertion fails loudly if any Phase 3 query touches the reserved TEST period (2024-07-01 to 2024-08-31).
"""

import pytest

from ingestion.db import DatabaseManager
from weighting.skill import (
    CALIBRATE_END,
    CALIBRATE_START,
    MIN_SAMPLE_THRESHOLD,
    SEASONS,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    assert_valid_skill_date_range,
    get_season_for_timestamp,
)
from weighting.weights import (
    compute_inverse_error_weights,
    WeightEngine,
)


def test_anti_leakage_assertion_strictly_guards_test_period():
    """Requirement 7c / Hard Rule 4: Leaked test split must fail loudly.
    
    Verifies that assert_valid_skill_date_range raises AssertionError
    whenever a date range touches or extends into the reserved TEST period.
    """
    # 1. Valid training / calibration dates should pass cleanly
    assert_valid_skill_date_range(TRAIN_START, TRAIN_END)
    assert_valid_skill_date_range(CALIBRATE_START, CALIBRATE_END)
    assert_valid_skill_date_range(TRAIN_START, CALIBRATE_END)

    # 2. Queries touching or extending into the reserved TEST period MUST raise AssertionError
    with pytest.raises(AssertionError, match="LEAKAGE VIOLATION"):
        assert_valid_skill_date_range("2024-01-01", TEST_START)

    with pytest.raises(AssertionError, match="LEAKAGE VIOLATION"):
        assert_valid_skill_date_range(CALIBRATE_START, TEST_END)

    with pytest.raises(AssertionError, match="LEAKAGE VIOLATION"):
        assert_valid_skill_date_range("2024-07-15", "2024-08-15")


def test_season_mapping_constants():
    """Verify that season definitions match explicit meteorological boundaries."""
    assert get_season_for_timestamp("2024-04-15T00:00") == "pre-monsoon"
    assert get_season_for_timestamp("2024-07-15T00:00") == "monsoon"
    assert get_season_for_timestamp("2024-10-15T00:00") == "post-monsoon"
    assert get_season_for_timestamp("2024-01-15T00:00") == "winter"


def test_weights_sum_to_one_across_various_distributions():
    """Requirement 7a: Normalized weights must strictly sum to 1.0."""
    models = ["ecmwf_ifs", "ecmwf_aifs", "gfs", "icon"]

    # Case 1: Diverse RMSE values
    rmse_map = {"ecmwf_ifs": 1.2, "ecmwf_aifs": 1.5, "gfs": 2.1, "icon": 1.8}
    sample_counts = {m: 100 for m in models}

    weights, is_low_conf = compute_inverse_error_weights(rmse_map, sample_counts, all_models=models)
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)
    assert is_low_conf is False

    # Best model (lowest RMSE) must receive the highest weight
    assert weights["ecmwf_ifs"] > weights["ecmwf_aifs"]
    assert weights["ecmwf_aifs"] > weights["icon"]
    assert weights["icon"] > weights["gfs"]


def test_low_sample_buckets_fallback_to_equal_weights():
    """Requirement 7b: Insufficient sample count (<30) must fall back to equal weights with low_confidence=True."""
    models = ["ecmwf_ifs", "ecmwf_aifs", "gfs", "icon"]

    # Only 10 samples (below MIN_SAMPLE_THRESHOLD = 30)
    rmse_map = {"ecmwf_ifs": 1.0, "ecmwf_aifs": 1.5, "gfs": 2.0, "icon": 2.5}
    low_sample_counts = {m: 10 for m in models}

    weights, is_low_conf = compute_inverse_error_weights(
        rmse_map,
        low_sample_counts,
        all_models=models,
        min_samples=MIN_SAMPLE_THRESHOLD,
    )

    # Must fall back to 0.25 equal weight for all models
    for m in models:
        assert weights[m] == pytest.approx(0.25)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert is_low_conf is True


def test_stored_database_weights_integrity():
    """Verify that all stored weights in SQLite sum to 1.0 for every bucket."""
    db = DatabaseManager()
    df = db.get_model_weights()

    if df.empty:
        pytest.skip("Model weights not yet generated in SQLite.")

    # Group by location, season, variable and check sum == 1.0
    for (loc, season, var), group in df.groupby(["location_id", "season", "variable"]):
        weight_sum = group["weight"].sum()
        assert weight_sum == pytest.approx(1.0, abs=1e-3), (
            f"Weights do not sum to 1.0 for ({loc}, {season}, {var}): sum={weight_sum}"
        )
