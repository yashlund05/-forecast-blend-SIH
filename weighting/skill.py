"""Historical Skill and Error Engine (Module 2).

Computes historical MAE and RMSE per (location, topography_class, season, lead_time_bucket, model)
strictly within the non-overlapping TRAIN + CALIBRATE time window.
Enforces Hard Rule 4 (zero data leakage into reserved TEST period).
"""

from datetime import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests

from ingestion.config import TARGET_LOCATIONS
from ingestion.db import DatabaseManager

logger = logging.getLogger(__name__)

# =====================================================================
# Hard Rule 4: Strictly Non-Overlapping Time Splits
# Leaked date ranges invalidate verification credibility.
# =====================================================================
TRAIN_START = "2023-09-01"
TRAIN_END = "2024-04-30"

CALIBRATE_START = "2024-05-01"
CALIBRATE_END = "2024-06-30"

# Reserved strictly for Phase 4 Verification & Phase 5 Extreme Case Study
# MUST REMAIN UNTOUCHED BY ANY PHASE 3 COMPUTATION
TEST_START = "2024-07-01"
TEST_END = "2024-08-31"

ALLOWED_SKILL_WINDOW = (TRAIN_START, CALIBRATE_END)

# Explicit season definitions (named constant, not implicit logic)
SEASON_MONTH_MAP: Dict[int, str] = {
    # Pre-monsoon: March to May
    3: "pre-monsoon",
    4: "pre-monsoon",
    5: "pre-monsoon",
    # Monsoon: June to September
    6: "monsoon",
    7: "monsoon",
    8: "monsoon",
    9: "monsoon",
    # Post-monsoon: October to November
    10: "post-monsoon",
    11: "post-monsoon",
    # Winter: December to February
    12: "winter",
    1: "winter",
    2: "winter",
}

SEASONS: List[str] = ["pre-monsoon", "monsoon", "post-monsoon", "winter"]

# Lead time bucket classifications
LEAD_TIME_BUCKETS: Dict[str, Tuple[int, int]] = {
    "short": (0, 48),  # Days 1-2
    "medium": (49, 120),  # Days 3-5
    "extended": (121, 168),  # Days 6-7
    "all": (0, 168),  # Aggregate
}

# Minimum observation samples required per bucket to trust empirical weights
MIN_SAMPLE_THRESHOLD = 30

# Power parameter for inverse-error weighting (w_m ∝ 1 / RMSE_m^p)
# p=2 is chosen because it penalizes models with large outlier errors (busts)
# significantly more sharply than linear p=1, rewarding models with consistent accuracy.
INVERSE_ERROR_POWER = 2.0

HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def assert_valid_skill_date_range(start_date: str, end_date: str) -> None:
    """Enforce strict guard against data leakage into the reserved TEST period.
    
    Raises AssertionError if any query attempts to access dates in or after TEST_START.
    """
    if start_date > end_date:
        raise ValueError(f"Invalid date range order: {start_date} > {end_date}")
    if end_date >= TEST_START:
        raise AssertionError(
            f"LEAKAGE VIOLATION: Skill computation queried date range up to {end_date}, "
            f"which overlaps with reserved TEST_START ({TEST_START}). "
            f"Allowed training+calibration window ends at {CALIBRATE_END}."
        )
    if start_date < TRAIN_START:
        raise AssertionError(
            f"Query start date {start_date} is earlier than TRAIN_START ({TRAIN_START})."
        )


def get_season_for_timestamp(dt_str: str) -> str:
    """Extract season name from ISO timestamp."""
    month = int(dt_str[5:7])
    return SEASON_MONTH_MAP.get(month, "all")


class SkillEngine:
    """Computes historical skill metrics (MAE, RMSE) for models against ground truth."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None) -> None:
        self.db = db_manager or DatabaseManager()

    def fetch_and_store_historical_forecasts(
        self,
        location_id: str,
        start_date: str = TRAIN_START,
        end_date: str = CALIBRATE_END,
    ) -> int:
        """Fetch historical model runs (GFS, ICON, ECMWF IFS) from Open-Meteo for training window.
        
        Strictly guarded against accessing the reserved TEST window.
        """
        assert_valid_skill_date_range(start_date, end_date)

        if location_id not in TARGET_LOCATIONS:
            raise ValueError(f"Unknown location: {location_id}")

        loc_cfg = TARGET_LOCATIONS[location_id]

        params = {
            "latitude": loc_cfg.latitude,
            "longitude": loc_cfg.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,precipitation,wind_speed_10m",
            "models": ["gfs_seamless", "icon_seamless", "ecmwf_ifs025"],
        }

        try:
            r = requests.get(HISTORICAL_FORECAST_URL, params=params, timeout=30)
            if r.status_code != 200:
                logger.error("Failed historical forecast fetch: HTTP %d %s", r.status_code, r.text[:200])
                return 0

            data = r.json().get("hourly", {})
            times = data.get("time", [])
            if not times:
                return 0

            records: List[Dict[str, Any]] = []
            models_to_extract = [
                ("gfs_seamless", "gfs"),
                ("icon_seamless", "icon"),
                ("ecmwf_ifs025", "ecmwf_ifs"),
            ]

            for idx, t_str in enumerate(times):
                for raw_m, canon_m in models_to_extract:
                    temp = data.get(f"temperature_2m_{raw_m}", [None])[idx]
                    precip = data.get(f"precipitation_{raw_m}", [None])[idx]
                    wind = data.get(f"wind_speed_10m_{raw_m}", [None])[idx]

                    if temp is not None:
                        records.append(
                            {
                                "location_id": location_id,
                                "target_time": t_str,
                                "model": canon_m,
                                "temperature_2m": float(temp),
                                "precipitation": float(precip) if precip is not None else 0.0,
                                "wind_speed_10m": float(wind) if wind is not None else 0.0,
                            }
                        )

            saved = self.db.save_historical_forecasts(records)
            logger.info("Saved %d historical forecast records for %s", saved, location_id)
            return saved

        except Exception as e:
            logger.error("Exception fetching historical forecasts for %s: %s", location_id, e)
            return 0

    def compute_skill_metrics(
        self,
        location_id: str,
        start_date: str = TRAIN_START,
        end_date: str = CALIBRATE_END,
    ) -> pd.DataFrame:
        """Compute MAE and RMSE per (location, season, lead_time_bucket, variable, model).
        
        Strictly enforces non-overlapping train+calibrate time range.
        """
        assert_valid_skill_date_range(start_date, end_date)

        loc_cfg = TARGET_LOCATIONS.get(location_id)
        if not loc_cfg:
            raise ValueError(f"Unknown location: {location_id}")

        # 1. Load ground truth reanalysis observations (strictly within train+calibrate window)
        obs_df = self.db.get_historical_observations(
            location_id=location_id,
            start_date=start_date,
            end_date=end_date,
        )
        if obs_df.empty:
            logger.warning("No observations found for %s between %s and %s", location_id, start_date, end_date)
            return pd.DataFrame()

        # 2. Load historical model forecasts
        fcst_df = self.db.get_historical_forecasts(
            location_id=location_id,
            start_date=start_date,
            end_date=end_date,
        )
        if fcst_df.empty:
            logger.info("Fetching historical forecasts for %s...", location_id)
            self.fetch_and_store_historical_forecasts(location_id, start_date, end_date)
            fcst_df = self.db.get_historical_forecasts(
                location_id=location_id,
                start_date=start_date,
                end_date=end_date,
            )

        if fcst_df.empty:
            logger.warning("No historical forecasts available for %s", location_id)
            return pd.DataFrame()

        # 3. Merge forecasts with observations on target_time == time
        merged = pd.merge(
            fcst_df,
            obs_df,
            left_on=["location_id", "target_time"],
            right_on=["location_id", "time"],
            suffixes=("_fcst", "_obs"),
        )

        if merged.empty:
            return pd.DataFrame()

        # Assign season column
        merged["season"] = [get_season_for_timestamp(t) for t in merged["target_time"]]

        variables = ["temperature_2m", "precipitation", "wind_speed_10m"]
        models = merged["model"].unique().tolist()
        seasons = merged["season"].unique().tolist()

        skill_records: List[Dict[str, Any]] = []

        # Compute metrics across overall lead time ("all") and seasons
        for season in seasons:
            s_sub = merged[merged["season"] == season]

            for var in variables:
                for model in models:
                    m_sub = s_sub[s_sub["model"] == model]
                    
                    y_fcst = m_sub[f"{var}_fcst"].to_numpy(dtype=float)
                    y_obs = m_sub[f"{var}_obs"].to_numpy(dtype=float)

                    valid_mask = ~np.isnan(y_fcst) & ~np.isnan(y_obs)
                    y_fcst = y_fcst[valid_mask]
                    y_obs = y_obs[valid_mask]
                    n_samples = len(y_fcst)

                    if n_samples > 0:
                        errors = y_fcst - y_obs
                        mae = float(np.mean(np.abs(errors)))
                        rmse = float(np.sqrt(np.mean(errors ** 2)))
                    else:
                        mae = np.nan
                        rmse = np.nan

                    skill_records.append(
                        {
                            "location_id": location_id,
                            "topography": loc_cfg.topography,
                            "season": season,
                            "lead_time_bucket": "all",
                            "variable": var,
                            "model": model,
                            "sample_count": n_samples,
                            "mae": round(mae, 3) if not np.isnan(mae) else None,
                            "rmse": round(rmse, 3) if not np.isnan(rmse) else None,
                        }
                    )

        return pd.DataFrame(skill_records)
