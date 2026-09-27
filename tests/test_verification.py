"""Unit tests for Verification Module (Module 5).

Covers:
- (a) Anti-leakage enforcement: Hard Rule 4 assertion fails loudly if verification touches TRAIN or CALIBRATE data.
- (b) Mathematical correctness of contingency scoring (POD, FAR, CSI, ETS).
- (c) Metrics computed from real stored database data (not hardcoded).
- (d) Hard Rule 5: Verification engine explicitly surfaces at least one honest failure case where the blend does not win.
"""

import numpy as np
import pytest

from verification.guards import (
    TEST_END,
    TEST_START,
    assert_strictly_test_period,
)
from verification.metrics import (
    ContingencyScores,
    VerificationEngine,
    compute_contingency_scores,
)


def test_verification_anti_leakage_guard():
    """Requirement 6a / Hard Rule 4: Verification code must ONLY touch the held-out TEST period."""
    # 1. Valid TEST period query should pass cleanly
    assert_strictly_test_period(TEST_START, TEST_END)
    assert_strictly_test_period("2024-07-15", "2024-08-15")

    # 2. Queries touching TRAIN or CALIBRATE data MUST raise AssertionError
    with pytest.raises(AssertionError, match="VERIFICATION LEAKAGE VIOLATION"):
        assert_strictly_test_period("2024-06-30", TEST_END)

    with pytest.raises(AssertionError, match="VERIFICATION LEAKAGE VIOLATION"):
        assert_strictly_test_period("2023-09-01", "2024-04-30")

    # 3. Queries beyond TEST_END must raise AssertionError
    with pytest.raises(AssertionError, match="VERIFICATION WINDOW VIOLATION"):
        assert_strictly_test_period(TEST_START, "2024-09-01")


def test_contingency_scores_math():
    """Verify exact formula implementations for POD, FAR, CSI, and ETS."""
    # Construct synthetic occurrences:
    # 5 Hits, 2 False Alarms, 3 Misses, 10 Correct Negatives (Total N = 20)
    y_true = np.array([1]*5 + [0]*2 + [1]*3 + [0]*10)
    y_pred = np.array([1]*5 + [1]*2 + [0]*3 + [0]*10)

    scores = compute_contingency_scores(y_true, y_pred, threshold=1)

    assert scores.hits == 5
    assert scores.false_alarms == 2
    assert scores.misses == 3
    assert scores.correct_negatives == 10

    # POD = H / (H + M) = 5 / 8 = 0.625
    assert scores.pod == pytest.approx(0.625)

    # FAR = FA / (H + FA) = 2 / 7 ≈ 0.286
    assert scores.far == pytest.approx(2 / 7, abs=1e-3)

    # CSI = H / (H + FA + M) = 5 / 10 = 0.500
    assert scores.csi == pytest.approx(0.500)

    # H_random = ((5 + 3) * (5 + 2)) / 20 = (8 * 7) / 20 = 2.8
    # ETS = (5 - 2.8) / (10 - 2.8) = 2.2 / 7.2 ≈ 0.306
    assert scores.ets == pytest.approx(2.2 / 7.2, abs=1e-3)


def test_metrics_computed_from_real_stored_data():
    """Requirement 6b: Verification metrics are computed from real stored reanalysis and forecast data."""
    engine = VerificationEngine()
    res = engine.evaluate_test_period(locations=["mumbai"])

    assert res["summary"]["total_stations_evaluated"] == 1
    assert res["summary"]["total_eval_points"] > 0

    df = res["metrics_table"]
    assert not df.empty
    assert set(df["variable"].unique()) == {"temperature_2m", "precipitation", "wind_speed_10m"}

    for _, row in df.iterrows():
        # Assert valid non-zero float numbers
        assert row["rmse_blend"] > 0
        assert row["rmse_naive"] > 0
        assert row["rmse_best_model"] > 0
        assert row["mae_blend"] > 0


def test_verification_surfaces_honest_failure_case():
    """Requirement 6c / Hard Rule 5: Verification must explicitly surface at least one real failure case."""
    engine = VerificationEngine()
    res = engine.evaluate_test_period(locations=["mumbai", "delhi"])

    failure_cases = res["failure_cases"]
    assert len(failure_cases) >= 1, "Must contain at least one honest failure case per Hard Rule 5."

    fc = failure_cases[0]
    assert "station_name" in fc
    assert "variable" in fc
    assert "rmse_blend" in fc
    assert "rmse_best_model" in fc
    assert "reason" in fc
    assert len(fc["reason"]) > 10
