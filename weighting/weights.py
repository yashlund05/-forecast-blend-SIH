"""Model Weight Generator and Lookup Engine (Module 2).

Converts empirical RMSE metrics into inverse-error weights (w_m ∝ 1 / RMSE_m^2).
Handles sample threshold enforcement, fallback to equal weighting on low confidence,
and SQLite persistence.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from blending.engine import BLEND_MODELS
from ingestion.config import TARGET_LOCATIONS
from ingestion.db import DatabaseManager
from weighting.skill import (
    INVERSE_ERROR_POWER,
    LEAD_TIME_BUCKETS,
    MIN_SAMPLE_THRESHOLD,
    SEASONS,
    SkillEngine,
)

logger = logging.getLogger(__name__)


def compute_inverse_error_weights(
    rmse_by_model: Dict[str, Optional[float]],
    sample_counts: Dict[str, int],
    all_models: Optional[List[str]] = None,
    power: float = INVERSE_ERROR_POWER,
    min_samples: int = MIN_SAMPLE_THRESHOLD,
) -> Tuple[Dict[str, float], bool]:
    """Compute normalized inverse-error weights: w_m ∝ 1 / (RMSE_m^p).
    
    Why p=2?
    Linear inverse weighting (p=1) treats a model with double the error as half as good.
    Quadratic inverse weighting (p=2) penalizes large variance and severe forecast busts
    proportionally to variance, rewarding consistently stable models.
    
    Returns:
        (weights_dict, is_low_confidence)
    """
    models = all_models or BLEND_MODELS
    n_models = len(models)
    equal_weight = 1.0 / n_models

    # Check sample counts
    has_low_samples = any(
        sample_counts.get(m, 0) < min_samples for m in models
    )

    # Filter models with valid positive RMSE and sufficient samples
    valid_inv_scores: Dict[str, float] = {}
    for m in models:
        cnt = sample_counts.get(m, 0)
        rmse = rmse_by_model.get(m)
        if cnt >= min_samples and rmse is not None and rmse > 0:
            valid_inv_scores[m] = 1.0 / (rmse ** power)

    if not valid_inv_scores:
        # Fallback to equal weighting if no model has sufficient samples
        return {m: equal_weight for m in models}, True

    total_inv = sum(valid_inv_scores.values())

    # If some models are missing samples (e.g. newly introduced AI model AIFS),
    # allocate a baseline equal share to uncalibrated models and distribute the rest
    # among calibrated models according to inverse error, marking low_confidence=True.
    weights: Dict[str, float] = {}
    if len(valid_inv_scores) < n_models:
        # Uncalibrated models get equal proportion, calibrated share the remainder
        uncalibrated_share = 0.25 * (n_models - len(valid_inv_scores))
        calibrated_share = 1.0 - uncalibrated_share

        for m in models:
            if m in valid_inv_scores:
                weights[m] = round(
                    calibrated_share * (valid_inv_scores[m] / total_inv), 4
                )
            else:
                weights[m] = 0.25
        is_low_confidence = True
    else:
        for m in models:
            weights[m] = round(valid_inv_scores[m] / total_inv, 4)
        is_low_confidence = has_low_samples

    # Final normalization to guarantee sum == 1.0
    s = sum(weights.values())
    if s > 0:
        weights = {m: round(w / s, 4) for m, w in weights.items()}

    return weights, is_low_confidence


class WeightEngine:
    """Generates, persists, and serves learned model weights."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        skill_engine: Optional[SkillEngine] = None,
    ) -> None:
        self.db = db_manager or DatabaseManager()
        self.skill_engine = skill_engine or SkillEngine(db_manager=self.db)

    def generate_and_save_weights(
        self,
        locations: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compute skill metrics and generate full weights table stored in SQLite."""
        target_keys = locations or list(TARGET_LOCATIONS.keys())
        updated_at = datetime.now(timezone.utc).isoformat()
        records_to_save: List[Dict[str, Any]] = []

        logger.info("Generating model weights for %d locations...", len(target_keys))

        for loc_id in target_keys:
            skill_df = self.skill_engine.compute_skill_metrics(location_id=loc_id)
            if skill_df.empty:
                logger.warning("No skill metrics generated for %s, skipping.", loc_id)
                continue

            variables = skill_df["variable"].unique()
            seasons = skill_df["season"].unique()

            for season in seasons:
                for var in variables:
                    sub = skill_df[
                        (skill_df["season"] == season) & (skill_df["variable"] == var)
                    ]

                    rmse_map = dict(zip(sub["model"], sub["rmse"]))
                    sample_map = dict(zip(sub["model"], sub["sample_count"]))
                    mae_map = dict(zip(sub["model"], sub["mae"]))

                    weights, is_low_conf = compute_inverse_error_weights(
                        rmse_by_model=rmse_map,
                        sample_counts=sample_map,
                        all_models=BLEND_MODELS,
                        power=INVERSE_ERROR_POWER,
                        min_samples=MIN_SAMPLE_THRESHOLD,
                    )

                    for model, w in weights.items():
                        records_to_save.append(
                            {
                                "location_id": loc_id,
                                "season": season,
                                "lead_time_bucket": "all",
                                "variable": var,
                                "model": model,
                                "weight": w,
                                "sample_count": sample_map.get(model, 0),
                                "rmse": rmse_map.get(model),
                                "mae": mae_map.get(model),
                                "low_confidence": 1 if is_low_conf else 0,
                                "updated_at": updated_at,
                            }
                        )

        if records_to_save:
            df_weights = pd.DataFrame(records_to_save)
            self.db.save_model_weights(df_weights)
            logger.info("Saved %d model weight entries to SQLite", len(records_to_save))
            return df_weights

        return pd.DataFrame()

    def get_weights_for_location(
        self,
        location_id: str,
        season: str,
        lead_time_bucket: str = "all",
    ) -> Dict[str, Dict[str, float]]:
        """Retrieve weights dictionary formatted as Dict[variable, Dict[model, weight]].
        
        Directly compatible with ForecastBlendEngine.blend(weights=...).
        """
        df = self.db.get_model_weights(
            location_id=location_id,
            season=season,
            lead_time_bucket=lead_time_bucket,
        )

        if df.empty:
            # Fallback to equal weighting
            equal = 1.0 / len(BLEND_MODELS)
            return {
                var: {m: equal for m in BLEND_MODELS}
                for var in ["temperature_2m", "precipitation", "wind_speed_10m"]
            }

        result: Dict[str, Dict[str, float]] = {}
        for var, group in df.groupby("variable"):
            result[var] = dict(zip(group["model"], group["weight"]))

        return result

    def get_dominant_model_map(
        self,
        season: str = "monsoon",
        lead_time_bucket: str = "all",
        variable: str = "temperature_2m",
    ) -> pd.DataFrame:
        """Return the highest-weighted model per station for map visualization."""
        df = self.db.get_model_weights(
            season=season,
            lead_time_bucket=lead_time_bucket,
            variable=variable,
        )

        rows: List[Dict[str, Any]] = []
        for loc_id, cfg in TARGET_LOCATIONS.items():
            loc_sub = df[df["location_id"] == loc_id] if not df.empty else pd.DataFrame()
            if not loc_sub.empty:
                best_row = loc_sub.loc[loc_sub["weight"].idxmax()]
                rows.append(
                    {
                        "location_id": loc_id,
                        "name": cfg.name,
                        "latitude": cfg.latitude,
                        "longitude": cfg.longitude,
                        "zone": cfg.zone,
                        "topography": cfg.topography,
                        "dominant_model": best_row["model"],
                        "weight": best_row["weight"],
                        "rmse": best_row["rmse"],
                        "low_confidence": bool(best_row["low_confidence"]),
                    }
                )
            else:
                # Default placeholder if weights not yet computed
                rows.append(
                    {
                        "location_id": loc_id,
                        "name": cfg.name,
                        "latitude": cfg.latitude,
                        "longitude": cfg.longitude,
                        "zone": cfg.zone,
                        "topography": cfg.topography,
                        "dominant_model": "equal_weight",
                        "weight": 0.25,
                        "rmse": None,
                        "low_confidence": True,
                    }
                )

        return pd.DataFrame(rows)
