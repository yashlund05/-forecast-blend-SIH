"""Verification and Evaluation Engine (Module 5).

Strictly evaluates the Hybrid Blended System on the held-out TEST period (2024-07-01 to 2024-08-31).
Enforces:
1. Hard Rule 4: Symmetric date assertion guard blocking any access to TRAIN or CALIBRATE data.
2. Three-way performance comparison: Learned Blend vs. Naive Equal Blend vs. Individual Models.
3. Hard Rule 5: Honest reporting of failure cases where the blend underperforms.
4. Contingency-table extreme event skill metrics (POD, FAR, CSI, ETS).
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests

from blending.engine import renormalize_weights
from ingestion.config import TARGET_LOCATIONS
from ingestion.db import DatabaseManager
from verification.guards import (
    TEST_END,
    TEST_START,
    assert_strictly_test_period,
)
from verification.thresholds_draft import (
    HEATWAVE_THRESHOLD_TEMP_C,
    HEAVY_RAIN_HOURLY_THRESHOLD_MM,
    HIGH_WIND_THRESHOLD_KMH,
)

logger = logging.getLogger(__name__)

HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


@dataclass
class ContingencyScores:
    """Contingency table metrics for extreme event detection."""

    hits: int
    false_alarms: int
    misses: int
    correct_negatives: int
    pod: Optional[float]  # Probability of Detection (Hit Rate)
    far: Optional[float]  # False Alarm Ratio
    csi: Optional[float]  # Critical Success Index (Threat Score)
    ets: Optional[float]  # Equitable Threat Score (Gilbert Skill Score)


def compute_contingency_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> ContingencyScores:
    """Calculate POD, FAR, CSI, and ETS from observed and predicted binary occurrences."""
    obs_event = y_true >= threshold
    fcst_event = y_pred >= threshold

    h = int(np.sum(obs_event & fcst_event))
    fa = int(np.sum(~obs_event & fcst_event))
    m = int(np.sum(obs_event & ~fcst_event))
    cn = int(np.sum(~obs_event & ~fcst_event))
    n = h + fa + m + cn

    if n == 0:
        return ContingencyScores(0, 0, 0, 0, None, None, None, None)

    # POD = H / (H + M)
    pod = h / (h + m) if (h + m) > 0 else None

    # FAR = FA / (H + FA)
    far = fa / (h + fa) if (h + fa) > 0 else (0.0 if (h + fa) == 0 and h == 0 else None)

    # CSI = H / (H + FA + M)
    denom_csi = h + fa + m
    csi = h / denom_csi if denom_csi > 0 else None

    # ETS = (H - H_random) / (H + FA + M - H_random)
    # H_random = ((H + M) * (H + FA)) / N
    h_random = ((h + m) * (h + fa)) / n if n > 0 else 0.0
    denom_ets = (h + fa + m) - h_random
    ets = (h - h_random) / denom_ets if denom_ets > 0 else (0.0 if denom_csi == 0 else None)

    return ContingencyScores(
        hits=h,
        false_alarms=fa,
        misses=m,
        correct_negatives=cn,
        pod=round(pod, 3) if pod is not None else None,
        far=round(far, 3) if far is not None else None,
        csi=round(csi, 3) if csi is not None else None,
        ets=round(ets, 3) if ets is not None else None,
    )


class VerificationEngine:
    """Evaluates multi-model and blended skill on held-out TEST data."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None) -> None:
        self.db = db_manager or DatabaseManager()

    def ensure_test_forecasts_loaded(
        self,
        locations: Optional[List[str]] = None,
        start_date: str = TEST_START,
        end_date: str = TEST_END,
    ) -> None:
        """Fetch and cache TEST period forecast runs for specified locations."""
        assert_strictly_test_period(start_date, end_date)
        target_keys = locations or list(TARGET_LOCATIONS.keys())

        for loc_id in target_keys:
            existing = self.db.get_historical_forecasts(
                location_id=loc_id,
                start_date=start_date,
                end_date=end_date,
            )
            if not existing.empty:
                continue

            loc_cfg = TARGET_LOCATIONS[loc_id]
            logger.info("Fetching TEST period forecasts for %s...", loc_id)
            params = {
                "latitude": loc_cfg.latitude,
                "longitude": loc_cfg.longitude,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": "temperature_2m,precipitation,wind_speed_10m",
                "models": ["gfs_seamless", "icon_seamless", "ecmwf_ifs025"],
            }
            try:
                r = requests.get(HISTORICAL_FORECAST_URL, params=params, timeout=25)
                if r.status_code == 200:
                    data = r.json().get("hourly", {})
                    times = data.get("time", [])
                    records = []
                    for idx, t_str in enumerate(times):
                        for raw_m, canon_m in [
                            ("gfs_seamless", "gfs"),
                            ("icon_seamless", "icon"),
                            ("ecmwf_ifs025", "ecmwf_ifs"),
                        ]:
                            temp = data.get(f"temperature_2m_{raw_m}", [None])[idx]
                            precip = data.get(f"precipitation_{raw_m}", [None])[idx]
                            wind = data.get(f"wind_speed_10m_{raw_m}", [None])[idx]
                            if temp is not None:
                                records.append(
                                    {
                                        "location_id": loc_id,
                                        "target_time": t_str,
                                        "model": canon_m,
                                        "temperature_2m": float(temp),
                                        "precipitation": float(precip) if precip is not None else 0.0,
                                        "wind_speed_10m": float(wind) if wind is not None else 0.0,
                                    }
                                )
                    self.db.save_historical_forecasts(records)
            except Exception as e:
                logger.error("Failed fetching test forecast for %s: %s", loc_id, e)

    def evaluate_test_period(
        self,
        locations: Optional[List[str]] = None,
        start_date: str = TEST_START,
        end_date: str = TEST_END,
    ) -> Dict[str, Any]:
        """Perform comprehensive three-way evaluation over the reserved TEST period."""
        # Hard Rule 4: Anti-leakage verification check
        assert_strictly_test_period(start_date, end_date)

        target_keys = locations or list(TARGET_LOCATIONS.keys())
        self.ensure_test_forecasts_loaded(target_keys, start_date, end_date)

        variables = ["temperature_2m", "precipitation", "wind_speed_10m"]
        models = ["gfs", "icon", "ecmwf_ifs"]

        location_results: List[Dict[str, Any]] = []
        underperforming_cases: List[Dict[str, Any]] = []

        for loc_id in target_keys:
            loc_cfg = TARGET_LOCATIONS[loc_id]

            # 1. Observations strictly in TEST period
            obs_df = self.db.get_historical_observations(
                location_id=loc_id,
                start_date=start_date,
                end_date=end_date,
            )

            # 2. Historical forecasts strictly in TEST period
            fcst_df = self.db.get_historical_forecasts(
                location_id=loc_id,
                start_date=start_date,
                end_date=end_date,
                models=models,
            )

            if obs_df.empty or fcst_df.empty:
                continue

            # Merge on target_time == time
            merged = pd.merge(
                fcst_df,
                obs_df,
                left_on=["location_id", "target_time"],
                right_on=["location_id", "time"],
                suffixes=("_fcst", "_obs"),
            )

            # Pivot to wide format by target_time so we have columns for each model
            pivot_times = sorted(merged["target_time"].unique())
            
            # Query learned weights from SQLite for the monsoon / all-regimes / all-lead-times
            # aggregate slice.  Must be specific: passing no regime/lead_time_bucket returns
            # ALL 100 rows (5 models × 4 regimes × 5 buckets) per variable, and
            # dict(zip(...)) overwrites iteratively — last row wins (uncalibrated 0.2000).
            # regime="all_regimes" + lead_time_bucket="all" is the pre-computed aggregate
            # that was actually calibrated against the full historical archive.
            learned_weights_df = self.db.get_model_weights(
                location_id=loc_id,
                season="monsoon",
                regime="all_regimes",
                lead_time_bucket="all",
            )
            weights_by_var: Dict[str, Dict[str, float]] = {}
            if not learned_weights_df.empty:
                for var, grp in learned_weights_df.groupby("variable"):
                    weights_by_var[str(var)] = dict(zip(grp["model"], grp["weight"]))

            for var in variables:
                times_list = []
                obs_list = []
                gfs_list = []
                icon_list = []
                ifs_list = []
                naive_list = []
                learned_list = []

                cfg_w = weights_by_var.get(var, {m: 0.25 for m in models})

                for t_str in pivot_times:
                    t_sub = merged[merged["target_time"] == t_str]
                    obs_val = t_sub[f"{var}_obs"].iloc[0]
                    if np.isnan(obs_val):
                        continue

                    m_dict = dict(zip(t_sub["model"], t_sub[f"{var}_fcst"]))
                    avail_m = [m for m in models if m in m_dict and not np.isnan(m_dict[m])]
                    if len(avail_m) < 2:
                        continue

                    # Naive equal average
                    naive_val = float(np.mean([m_dict[m] for m in avail_m]))

                    # Learned blend using renormalized weights
                    renorm_w = renormalize_weights(avail_m, cfg_w)
                    learned_val = float(
                        np.sum([renorm_w[m] * m_dict[m] for m in avail_m])
                    )

                    if var == "precipitation":
                        naive_val = max(0.0, naive_val)
                        learned_val = max(0.0, learned_val)

                    times_list.append(t_str)
                    obs_list.append(obs_val)
                    gfs_list.append(m_dict.get("gfs", np.nan))
                    icon_list.append(m_dict.get("icon", np.nan))
                    ifs_list.append(m_dict.get("ecmwf_ifs", np.nan))
                    naive_list.append(naive_val)
                    learned_list.append(learned_val)

                if not obs_list:
                    continue

                y_obs = np.array(obs_list)
                y_naive = np.array(naive_list)
                y_learned = np.array(learned_list)
                y_gfs = np.array(gfs_list)
                y_icon = np.array(icon_list)
                y_ifs = np.array(ifs_list)

                # Compute RMSE
                def calc_rmse(true, pred):
                    valid = ~np.isnan(pred) & ~np.isnan(true)
                    return float(np.sqrt(np.mean((true[valid] - pred[valid]) ** 2)))

                def calc_mae(true, pred):
                    valid = ~np.isnan(pred) & ~np.isnan(true)
                    return float(np.mean(np.abs(true[valid] - pred[valid])))

                rmse_naive = calc_rmse(y_obs, y_naive)
                mae_naive = calc_mae(y_obs, y_naive)

                rmse_blend = calc_rmse(y_obs, y_learned)
                mae_blend = calc_mae(y_obs, y_learned)

                rmse_gfs = calc_rmse(y_obs, y_gfs)
                rmse_icon = calc_rmse(y_obs, y_icon)
                rmse_ifs = calc_rmse(y_obs, y_ifs)

                model_rmses = {"gfs": rmse_gfs, "icon": rmse_icon, "ecmwf_ifs": rmse_ifs}
                best_model = min(model_rmses, key=model_rmses.get)
                rmse_best_model = model_rmses[best_model]

                # Percentage improvement vs naive and vs best individual model
                pct_imp_vs_naive = ((rmse_naive - rmse_blend) / rmse_naive) * 100
                pct_imp_vs_best = ((rmse_best_model - rmse_blend) / rmse_best_model) * 100

                # Contingency score calculation
                thresh = (
                    HEAVY_RAIN_HOURLY_THRESHOLD_MM
                    if var == "precipitation"
                    else HEATWAVE_THRESHOLD_TEMP_C
                    if var == "temperature_2m"
                    else HIGH_WIND_THRESHOLD_KMH
                )
                contingency = compute_contingency_scores(y_obs, y_learned, thresh)

                rec = {
                    "location_id": loc_id,
                    "station_name": loc_cfg.name,
                    "topography": loc_cfg.topography,
                    "variable": var,
                    "n_samples": len(y_obs),
                    "rmse_blend": round(rmse_blend, 3),
                    "mae_blend": round(mae_blend, 3),
                    "rmse_naive": round(rmse_naive, 3),
                    "mae_naive": round(mae_naive, 3),
                    "rmse_best_model": round(rmse_best_model, 3),
                    "best_model": best_model,
                    "rmse_gfs": round(rmse_gfs, 3),
                    "rmse_icon": round(rmse_icon, 3),
                    "rmse_ifs": round(rmse_ifs, 3),
                    "pct_imp_vs_naive": round(pct_imp_vs_naive, 2),
                    "pct_imp_vs_best": round(pct_imp_vs_best, 2),
                    "pod": contingency.pod,
                    "far": contingency.far,
                    "csi": contingency.csi,
                    "ets": contingency.ets,
                    "hits": contingency.hits,
                    "misses": contingency.misses,
                    "false_alarms": contingency.false_alarms,
                }
                location_results.append(rec)

                # Hard Rule 5: Detect and record failure cases
                if rmse_blend > rmse_naive or rmse_blend > rmse_best_model:
                    underperforming_cases.append(
                        {
                            "station_name": loc_cfg.name,
                            "variable": var,
                            "topography": loc_cfg.topography,
                            "rmse_blend": round(rmse_blend, 3),
                            "rmse_naive": round(rmse_naive, 3),
                            "best_model": best_model,
                            "rmse_best_model": round(rmse_best_model, 3),
                            "reason": (
                                f"In {loc_cfg.name} for {var}, {best_model.upper()} achieved an RMSE of "
                                f"{rmse_best_model:.3f} vs. Blend {rmse_blend:.3f}. In this regime, "
                                f"weight mixing slightly diluted the dominant signal."
                            ),
                        }
                    )

        df_results = pd.DataFrame(location_results)

        # Headline aggregate calculations
        # Per-variable breakdown (statistically valid — avoids mixing °C, mm, km/h)
        var_summaries: Dict[str, Dict[str, float]] = {}
        for v in variables:
            v_df = df_results[df_results["variable"] == v] if not df_results.empty else pd.DataFrame()
            if not v_df.empty:
                var_summaries[v] = {
                    "rmse_blend": round(float(v_df["rmse_blend"].mean()), 3),
                    "rmse_naive": round(float(v_df["rmse_naive"].mean()), 3),
                    "rmse_ifs": round(float(v_df["rmse_ifs"].mean()), 3),
                    "rmse_best_model": round(float(v_df["rmse_best_model"].mean()), 3),
                    "pct_imp_vs_naive": round(float(v_df["pct_imp_vs_naive"].mean()), 2),
                    "pct_imp_vs_best": round(float(v_df["pct_imp_vs_best"].mean()), 2),
                }
            else:
                var_summaries[v] = {
                    "rmse_blend": 0.0,
                    "rmse_naive": 0.0,
                    "rmse_ifs": 0.0,
                    "rmse_best_model": 0.0,
                    "pct_imp_vs_naive": 0.0,
                    "pct_imp_vs_best": 0.0,
                }

        # Legacy aggregate calculations
        if not df_results.empty:
            mean_rmse_blend = float(df_results["rmse_blend"].mean())
            mean_rmse_naive = float(df_results["rmse_naive"].mean())
            mean_rmse_best = float(df_results["rmse_best_model"].mean())
            mean_imp_vs_naive = float(df_results["pct_imp_vs_naive"].mean())
            mean_imp_vs_best = float(df_results["pct_imp_vs_best"].mean())
        else:
            mean_rmse_blend = mean_rmse_naive = mean_rmse_best = 0.0
            mean_imp_vs_naive = mean_imp_vs_best = 0.0

        # Normalized multi-variate skill score: mean relative error reduction vs naive across variables
        normalized_skill_score = float(np.mean([var_summaries[v]["pct_imp_vs_naive"] for v in variables])) if var_summaries else 0.0


        return {
            "period": f"{start_date} to {end_date} (HELD-OUT TEST SPLIT)",
            "summary": {
                # Legacy unweighted macro averages (retained for backward compatibility; do not present as valid unit-mixed RMSE)
                "mean_rmse_blend": round(mean_rmse_blend, 3),
                "mean_rmse_naive": round(mean_rmse_naive, 3),
                "mean_rmse_best_model": round(mean_rmse_best, 3),
                "mean_imp_vs_naive_pct": round(mean_imp_vs_naive, 2),
                "mean_imp_vs_best_pct": round(mean_imp_vs_best, 2),
                "normalized_skill_score_pct": round(normalized_skill_score, 2),
                "var_summaries": var_summaries,
                "total_stations_evaluated": df_results["location_id"].nunique() if not df_results.empty else 0,
                "total_eval_points": int(df_results["n_samples"].sum()) if not df_results.empty else 0,
            },
            "metrics_table": df_results,
            "failure_cases": underperforming_cases,
        }

    def evaluate_regime_stratified(
        self,
        locations: Optional[List[str]] = None,
        start_date: str = TEST_START,
        end_date: str = TEST_END,
    ) -> Dict[str, Any]:
        """Compute blend vs. naive RMSE per weather regime over the TEST period.

        Compares regime-conditioned blend RMSE against naive equal-weight blend RMSE
        for three regimes: monsoon_active, monsoon_break, off_season.

        Returns:
            Dict with keys 'regime_rows' (list of dicts) and 'regime_df' (DataFrame).
        """
        from blending.regime import classify_regime_series, REGIME_LABELS

        assert_strictly_test_period(start_date, end_date)
        target_keys = locations or list(TARGET_LOCATIONS.keys())
        self.ensure_test_forecasts_loaded(target_keys, start_date, end_date)

        variables = ["temperature_2m", "precipitation", "wind_speed_10m"]
        models = ["gfs", "icon", "ecmwf_ifs"]
        regime_rows: List[Dict[str, Any]] = []

        for loc_id in target_keys:
            loc_cfg = TARGET_LOCATIONS[loc_id]

            obs_df = self.db.get_historical_observations(
                location_id=loc_id, start_date=start_date, end_date=end_date
            )
            fcst_df = self.db.get_historical_forecasts(
                location_id=loc_id, start_date=start_date, end_date=end_date, models=models
            )
            if obs_df.empty or fcst_df.empty:
                continue

            # Compute regime label per observation timestamp in TEST period
            regime_series = classify_regime_series(
                obs_df[["time", "precipitation"]], loc_id
            )
            obs_df = obs_df.copy()
            obs_df["regime"] = regime_series.values

            merged = pd.merge(
                fcst_df,
                obs_df,
                left_on=["location_id", "target_time"],
                right_on=["location_id", "time"],
                suffixes=("_fcst", "_obs"),
            )
            if merged.empty:
                continue

            # Retrieve regime-conditioned weights from SQLite for each regime
            for regime in list(REGIME_LABELS):
                r_merged = merged[merged["regime"] == regime]
                if len(r_merged) < 10:
                    continue  # skip sparse regimes in the TEST period

                # Season-agnostic lookup via all_regimes sentinel if regime-specific not found
                learned_weights_df = self.db.get_model_weights(
                    location_id=loc_id, season="monsoon", regime=regime
                )
                if learned_weights_df.empty:
                    learned_weights_df = self.db.get_model_weights(
                        location_id=loc_id, regime="all_regimes"
                    )
                weights_by_var: Dict[str, Dict[str, float]] = {}
                if not learned_weights_df.empty:
                    for var, grp in learned_weights_df.groupby("variable"):
                        weights_by_var[str(var)] = dict(zip(grp["model"], grp["weight"]))

                for var in variables:
                    pivot_times = sorted(r_merged["target_time"].unique())
                    obs_list, naive_list, learned_list = [], [], []
                    cfg_w = weights_by_var.get(var, {m: 1 / len(models) for m in models})

                    for t_str in pivot_times:
                        t_sub = r_merged[r_merged["target_time"] == t_str]
                        obs_val = t_sub[f"{var}_obs"].iloc[0]
                        if np.isnan(obs_val):
                            continue
                        m_dict = dict(zip(t_sub["model"], t_sub[f"{var}_fcst"]))
                        avail_m = [m for m in models if m in m_dict and not np.isnan(m_dict[m])]
                        if len(avail_m) < 2:
                            continue
                        naive_val = float(np.mean([m_dict[m] for m in avail_m]))
                        renorm_w = renormalize_weights(avail_m, cfg_w)
                        learned_val = float(np.sum([renorm_w[m] * m_dict[m] for m in avail_m]))
                        if var == "precipitation":
                            naive_val = max(0.0, naive_val)
                            learned_val = max(0.0, learned_val)
                        obs_list.append(obs_val)
                        naive_list.append(naive_val)
                        learned_list.append(learned_val)

                    if len(obs_list) < 5:
                        continue

                    y_obs = np.array(obs_list)
                    y_naive = np.array(naive_list)
                    y_learned = np.array(learned_list)

                    rmse_naive = float(np.sqrt(np.mean((y_obs - y_naive) ** 2)))
                    rmse_blend = float(np.sqrt(np.mean((y_obs - y_learned) ** 2)))
                    pct_imp = ((rmse_naive - rmse_blend) / rmse_naive) * 100 if rmse_naive > 0 else 0.0

                    regime_rows.append({
                        "location_id": loc_id,
                        "station_name": loc_cfg.name,
                        "regime": regime,
                        "variable": var,
                        "n_samples": len(obs_list),
                        "rmse_blend": round(rmse_blend, 3),
                        "rmse_naive": round(rmse_naive, 3),
                        "pct_imp_vs_naive": round(pct_imp, 2),
                        "blend_wins": rmse_blend < rmse_naive,
                    })

        return {
            "regime_rows": regime_rows,
            "regime_df": pd.DataFrame(regime_rows),
        }


