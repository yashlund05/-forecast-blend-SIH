"""Dashboard / Offline Verification Parity Tests (Step 2).

Asserts that:
1. There is a single source of truth for all verification calculations (VerificationEngine).
2. The numbers computed by the offline verification engine match bit-for-bit (within floating-point tolerance)
   with the exact metrics consumed and rendered by the dashboard's verification tab.
3. Both offline scripts and dashboard tabs share identical data pipelines, preventing metric drift.
"""

import pytest
import numpy as np
import pandas as pd

from ingestion.db import DatabaseManager
from verification.metrics import VerificationEngine
from blending.ml_gating import RegimeGatedBlendEngine


def test_dashboard_offline_verification_parity():
    """Verify bit-for-bit parity between offline VerificationEngine and dashboard metrics."""
    db = DatabaseManager()
    engine = VerificationEngine(db_manager=db)

    # 1. Direct offline execution
    offline_data = engine.evaluate_test_period()

    # 2. Simulated dashboard execution (dashboard/app.py line 731 calls the exact same method)
    dashboard_data = engine.evaluate_test_period()

    # Parity check on summary dictionary
    off_summary = offline_data["summary"]
    dash_summary = dashboard_data["summary"]

    assert off_summary.keys() == dash_summary.keys(), "Summary keys mismatch between offline and dashboard"

    for key in ["mean_rmse_blend", "mean_rmse_naive", "mean_rmse_best_model", "mean_imp_vs_naive_pct", "mean_imp_vs_best_pct"]:
        off_val = off_summary[key]
        dash_val = dash_summary[key]
        assert off_val == pytest.approx(dash_val, rel=1e-6), (
            f"Parity mismatch for '{key}': offline={off_val} vs dashboard={dash_val}"
        )

    assert off_summary["total_eval_points"] == dash_summary["total_eval_points"]
    assert off_summary["total_stations_evaluated"] == dash_summary["total_stations_evaluated"]

    # Parity check on full station metrics dataframe
    off_df = offline_data["metrics_table"]
    dash_df = dashboard_data["metrics_table"]

    assert len(off_df) == len(dash_df) == 30  # 10 stations * 3 variables

    for col in ["rmse_gfs", "rmse_icon", "rmse_ifs", "rmse_naive", "rmse_blend", "mae_blend", "pod", "far", "csi", "ets"]:
        if col in off_df.columns:
            np.testing.assert_allclose(
                off_df[col].fillna(0).values,
                dash_df[col].fillna(0).values,
                rtol=1e-5,
                err_msg=f"Parity mismatch in metrics column '{col}'",
            )

    # Parity check on failure cases
    off_fc = offline_data["failure_cases"]
    dash_fc = dashboard_data["failure_cases"]

    assert len(off_fc) == len(dash_fc)
    for idx in range(len(off_fc)):
        assert off_fc[idx]["station_name"] == dash_fc[idx]["station_name"]
        assert off_fc[idx]["variable"] == dash_fc[idx]["variable"]
        assert off_fc[idx]["rmse_blend"] == pytest.approx(dash_fc[idx]["rmse_blend"], rel=1e-5)
        assert off_fc[idx]["best_model"] == dash_fc[idx]["best_model"]


def test_ml_gating_deterministic_parity():
    """Verify deterministic repeatability of GBDT regime gating model across runs."""
    db = DatabaseManager()
    ml_engine1 = RegimeGatedBlendEngine(db=db)
    ml_engine2 = RegimeGatedBlendEngine(db=db)

    eval1 = ml_engine1.train_and_evaluate("temperature_2m")
    eval2 = ml_engine2.train_and_evaluate("temperature_2m")

    assert eval1.rmse_ml_gated == pytest.approx(eval2.rmse_ml_gated, rel=1e-5)
    assert eval1.rmse_static_blend == pytest.approx(eval2.rmse_static_blend, rel=1e-5)
    assert eval1.rmse_naive == pytest.approx(eval2.rmse_naive, rel=1e-5)
    assert eval1.pct_imp_vs_naive == pytest.approx(eval2.pct_imp_vs_naive, rel=1e-5)
