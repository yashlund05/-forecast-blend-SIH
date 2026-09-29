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
    interpolate_weight_grid,
    mask_grid_to_india,
    MAX_INTERPOLATION_DISTANCE_KM,
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

    # Group by location, season, regime, lead_time_bucket, variable and check sum == 1.0
    group_cols = ["location_id", "season", "lead_time_bucket", "variable"]
    if "regime" in df.columns:
        group_cols = ["location_id", "season", "regime", "lead_time_bucket", "variable"]

    for key, group in df.groupby(group_cols):
        weight_sum = group["weight"].sum()
        assert weight_sum == pytest.approx(1.0, abs=1e-3), (
            f"Weights do not sum to 1.0 for {key}: sum={weight_sum}"
        )



def test_weights_differ_across_lead_time_buckets():
    """Requirement 4: Assert weights for the same location/season differ across at least two lead-time buckets.
    
    Guards against lead-time bucketing silently collapsing to uniform/static values.
    """
    db = DatabaseManager()
    df = db.get_model_weights(location_id="mumbai", season="pre-monsoon", variable="precipitation")
    if df.empty:
        pytest.skip("Model weights not yet generated in SQLite.")

    # Compare 0-24h vs 24-72h weights
    w_0_24 = df[df["lead_time_bucket"] == "0-24h"].set_index("model")["weight"].to_dict()
    w_24_72 = df[df["lead_time_bucket"] == "24-72h"].set_index("model")["weight"].to_dict()

    assert len(w_0_24) > 0, "No weights found for lead_time_bucket '0-24h'"
    assert len(w_24_72) > 0, "No weights found for lead_time_bucket '24-72h'"

    # Check that at least one model has a non-zero weight difference across buckets
    differences = [abs(w_0_24[m] - w_24_72[m]) for m in w_0_24 if m in w_24_72]
    max_diff = max(differences)
    assert max_diff > 0.01, f"Weights across lead-time buckets 0-24h and 24-72h must differ, max_diff={max_diff}"


def test_weight_explainability_trace():
    """Verify that WeightExplainabilityEngine produces complete mathematical trace and physical rationale."""
    from weighting.explainability import WeightExplainabilityEngine

    engine = WeightExplainabilityEngine()
    trace = engine.explain_weights(location_id="mumbai", season="monsoon", variable="precipitation", lead_time_bucket="0-24h")

    assert trace.location_id == "mumbai"
    assert trace.station_name == "Mumbai"
    assert trace.topography == "coastal"
    assert trace.lead_time_bucket == "0-24h"
    assert "SUM" in trace.formula
    assert len(trace.models) > 0
    assert "Maritime boundary layer" in trace.meteorological_rationale
    for m in trace.models:
        assert "model" in m
        assert "final_weight" in m
        assert "rmse" in m


def test_regime_classifier_known_cases():
    """Verify classify_regime returns correct labels for constructed known scenarios.

    Uses synthetic DataFrames to avoid depending on live DB state.
    """
    import pandas as pd
    from blending.regime import classify_regime

    # Case 1: 7-day heavy rainfall in June (monsoon window) → monsoon_active
    times = pd.date_range("2023-06-01", periods=200, freq="h")
    obs_active = pd.DataFrame({
        "time": times.strftime("%Y-%m-%dT%H:%M"),
        "precipitation": [1.0] * 200,  # 1 mm/hr well above 0.3 threshold
    })
    assert classify_regime(obs_active, "mumbai") == "monsoon_active", \
        "Expected monsoon_active during June with heavy precipitation"

    # Case 2: dry spell in July (monsoon window) → monsoon_break
    obs_break = pd.DataFrame({
        "time": times.strftime("%Y-%m-%dT%H:%M"),
        "precipitation": [0.0] * 200,  # 0 mm/hr → break
    })
    assert classify_regime(obs_break, "mumbai") == "monsoon_break", \
        "Expected monsoon_break during July with zero precipitation"

    # Case 3: January (off-season for all SW-monsoon stations) → off_season
    times_jan = pd.date_range("2024-01-10", periods=200, freq="h")
    obs_off = pd.DataFrame({
        "time": times_jan.strftime("%Y-%m-%dT%H:%M"),
        "precipitation": [0.5] * 200,  # even heavy rain in Jan → off_season
    })
    assert classify_regime(obs_off, "mumbai") == "off_season", \
        "Expected off_season in January regardless of precipitation"

    # Case 4: October for Chennai (SE monsoon station) with heavy rain → monsoon_active
    times_oct = pd.date_range("2023-10-01", periods=200, freq="h")
    obs_chennai_oct = pd.DataFrame({
        "time": times_oct.strftime("%Y-%m-%dT%H:%M"),
        "precipitation": [1.5] * 200,
    })
    assert classify_regime(obs_chennai_oct, "chennai") == "monsoon_active", \
        "Expected monsoon_active for Chennai in October (SE monsoon window)"


def test_regime_weights_differ_from_all_regimes():
    """Verify that regime-stratified weights differ from the all_regimes aggregate.

    Guards against regime conditioning silently collapsing to the season aggregate.
    """
    db = DatabaseManager()
    if db.get_model_weights(location_id="mumbai", season="monsoon", regime="monsoon_active").empty:
        pytest.skip("Regime weights not yet generated in SQLite.")

    df_active = db.get_model_weights(
        location_id="mumbai", season="monsoon",
        regime="monsoon_active", variable="temperature_2m", lead_time_bucket="all"
    )
    df_all = db.get_model_weights(
        location_id="mumbai", season="monsoon",
        regime="all_regimes", variable="temperature_2m", lead_time_bucket="all"
    )

    assert not df_active.empty, "monsoon_active weights missing for mumbai/monsoon/temperature_2m"
    assert not df_all.empty, "all_regimes weights missing for mumbai/monsoon/temperature_2m"

    w_active = df_active.set_index("model")["weight"].to_dict()
    w_all = df_all.set_index("model")["weight"].to_dict()

    # At least one model's weight must differ by > 0.005 across regimes
    common_models = set(w_active) & set(w_all)
    diffs = [abs(w_active[m] - w_all[m]) for m in common_models]
    assert max(diffs) > 0.005, (
        f"Regime-stratified weights must differ from all_regimes aggregate. "
        f"active={w_active}, all={w_all}"
    )


def test_regime_weights_sum_to_one():
    """Verify weights sum to 1.0 across all regime × season × lead_time × variable cells."""
    db = DatabaseManager()
    df = db.get_model_weights(location_id="mumbai")
    if df.empty:
        pytest.skip("Model weights not yet generated in SQLite.")

    if "regime" not in df.columns:
        pytest.skip("Regime column not present — run generate_and_save_weights first.")

    for (loc, season, regime, lead_b, var), group in df.groupby(
        ["location_id", "season", "regime", "lead_time_bucket", "variable"]
    ):
        weight_sum = group["weight"].sum()
        assert weight_sum == pytest.approx(1.0, abs=1e-3), (
            f"Weights do not sum to 1.0 for ({loc}, {season}, {regime}, {lead_b}, {var}): "
            f"sum={weight_sum}"
        )


def test_interpolate_weight_grid_smoke():
    """Smoke test for coarse spatial IDW interpolation of station weights onto national grid.
    
    Verifies that interpolate_weight_grid:
    1. Operates without crashing on known small input (e.g. 3 benchmark points).
    2. Generates correct 2D output grid shape.
    3. Produces strictly finite, positive values bounded within the input weight range.
    4. Accurately reproduces known values at exact station coordinates.
    """
    import numpy as np

    known_lats = np.array([19.076, 28.613, 13.082])  # Mumbai, Delhi, Chennai
    known_lons = np.array([72.877, 77.209, 80.270])
    known_weights = np.array([0.45, 0.20, 0.35])

    grid_lats, grid_lons, interp_grid = interpolate_weight_grid(
        station_lats=known_lats,
        station_lons=known_lons,
        station_values=known_weights,
        grid_lat_min=10.0,
        grid_lat_max=32.0,
        grid_lon_min=70.0,
        grid_lon_max=85.0,
        n_points_lat=10,
        n_points_lon=10,
        power=2.0,
    )

    # Output shape checks
    assert len(grid_lats) == 10
    assert len(grid_lons) == 10
    assert interp_grid.shape == (10, 10)

    # Numerical validity: all points must be finite and within [min, max]
    assert np.all(np.isfinite(interp_grid)), "Interpolated grid contains NaN or Inf values"
    assert np.min(interp_grid) >= np.min(known_weights) - 1e-4, "Grid value below min weight"
    assert np.max(interp_grid) <= np.max(known_weights) + 1e-4, "Grid value above max weight"

    # Exact station coordinate test
    exact_lats, exact_lons, exact_grid = interpolate_weight_grid(
        station_lats=known_lats,
        station_lons=known_lons,
        station_values=known_weights,
        grid_lat_min=19.076,
        grid_lat_max=19.076,
        grid_lon_min=72.877,
        grid_lon_max=72.877,
        n_points_lat=1,
        n_points_lon=1,
    )
    assert exact_grid[0, 0] == pytest.approx(0.45, abs=1e-3)


def test_mask_grid_to_india_known_locations():
    """Verify that points outside India are masked out while points inside India are preserved.

    Requirement:
    - Known point outside India (e.g. Mid Arabian Sea at 15.0°N, 65.0°E) must be masked to NaN.
    - Known point outside India (e.g. Bay of Bengal at 15.0°N, 88.0°E) must be masked to NaN.
    - Known point inside India (e.g. Nagpur at 21.1458°N, 79.0882°E) must be kept valid.
    """
    import numpy as np

    # Benchmark test points: Nagpur (inside), Arabian Sea (outside), Bay of Bengal (outside)
    # Using dummy stations near Nagpur so proximity filter is satisfied for Nagpur
    station_lats = np.array([21.1458, 19.0760])  # Nagpur, Mumbai
    station_lons = np.array([79.0882, 72.8777])

    # 1. Test point inside India: Nagpur (21.1458, 79.0882)
    grid_lats_in = np.array([21.1458])
    grid_lons_in = np.array([79.0882])
    interp_in = np.array([[0.40]])

    masked_in = mask_grid_to_india(
        grid_lats=grid_lats_in,
        grid_lons=grid_lons_in,
        interp_grid=interp_in,
        station_lats=station_lats,
        station_lons=station_lons,
        max_distance_km=500.0,
    )
    assert not np.isnan(masked_in[0, 0]), "Expected Nagpur (inside India) to be preserved, but was masked out."
    assert masked_in[0, 0] == pytest.approx(0.40)

    # 2. Test point outside India: Mid Arabian Sea (15.0°N, 65.0°E)
    grid_lats_sea = np.array([15.0])
    grid_lons_sea = np.array([65.0])
    interp_sea = np.array([[0.35]])

    masked_sea = mask_grid_to_india(
        grid_lats=grid_lats_sea,
        grid_lons=grid_lons_sea,
        interp_grid=interp_sea,
        station_lats=station_lats,
        station_lons=station_lons,
        max_distance_km=2000.0,  # Even with infinite distance allowance, sovereign mask must discard
    )
    assert np.isnan(masked_sea[0, 0]), "Expected Arabian Sea (outside India) to be masked to NaN."

    # 3. Test point outside India: Central Bay of Bengal (15.0°N, 88.0°E)
    grid_lats_bob = np.array([15.0])
    grid_lons_bob = np.array([88.0])
    interp_bob = np.array([[0.50]])

    masked_bob = mask_grid_to_india(
        grid_lats=grid_lats_bob,
        grid_lons=grid_lons_bob,
        interp_grid=interp_bob,
        station_lats=station_lats,
        station_lons=station_lons,
        max_distance_km=2000.0,
    )
    assert np.isnan(masked_bob[0, 0]), "Expected Bay of Bengal (outside India) to be masked to NaN."


def test_mask_grid_to_india_distance_cutoff():
    """Verify that cells exceeding MAX_INTERPOLATION_DISTANCE_KM are masked even if inside India."""
    import numpy as np

    # Place a single station at Thiruvananthapuram (southern tip: 8.52°N, 76.94°E)
    # Test a point at Delhi (28.61°N, 77.21°E, inside India, ~2200 km away)
    station_lats = np.array([8.5241])
    station_lons = np.array([76.9366])

    grid_lats = np.array([28.6139])  # Delhi
    grid_lons = np.array([77.2090])
    interp = np.array([[0.33]])

    # With 500 km cutoff, Delhi should be masked out because nearest station is > 2000 km away
    masked = mask_grid_to_india(
        grid_lats=grid_lats,
        grid_lons=grid_lons,
        interp_grid=interp,
        station_lats=station_lats,
        station_lons=station_lons,
        max_distance_km=500.0,
    )
    assert np.isnan(masked[0, 0]), "Expected remote cell > 500 km from station to be masked to NaN."

    # With 3000 km cutoff, Delhi should be kept because it is inside India
    kept = mask_grid_to_india(
        grid_lats=grid_lats,
        grid_lons=grid_lons,
        interp_grid=interp,
        station_lats=station_lats,
        station_lons=station_lons,
        max_distance_km=3000.0,
    )
    assert not np.isnan(kept[0, 0]), "Expected cell within extended cutoff to be preserved inside India."


