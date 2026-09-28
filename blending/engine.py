"""Hybrid Forecast Blending Engine (Module 3).

Dynamically combines multi-model physical NWP and AI forecasts using
weighted averaging with automatic weight renormalization for missing/degraded sources.
Pluggable weight interface allows drop-in learned weights from Module 2 (Skill/Weight Engine).
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical model identifiers participating in the blend (3 NWP + 2 AI/ML)
BLEND_MODELS: List[str] = [
    "ecmwf_ifs",
    "ecmwf_aifs",
    "weathernext",
    "gfs",
    "icon",
]

# Forecast variables to blend
BLEND_VARIABLES: List[str] = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
]

# Default equal weights across all 5 models (0.20 each)
DEFAULT_MODEL_WEIGHTS: Dict[str, float] = {
    model: 1.0 / len(BLEND_MODELS) for model in BLEND_MODELS
}


def renormalize_weights(
    available_models: List[str],
    configured_weights: Dict[str, float],
) -> Dict[str, float]:
    """Renormalize weights among currently available models so they sum to 1.0.
    
    If none of the configured models are found or total weight is zero,
    falls back to equal weighting among available models.
    """
    if not available_models:
        return {}

    sub_weights = {
        m: max(0.0, configured_weights.get(m, 0.0))
        for m in available_models
    }
    total_w = sum(sub_weights.values())

    if total_w <= 0.0:
        # Fallback to uniform distribution among available sources
        uniform = 1.0 / len(available_models)
        return {m: uniform for m in available_models}

    return {m: w / total_w for m, w in sub_weights.items()}


@dataclass
class BlendedForecastResult:
    """Encapsulates blended forecast dataframe and metadata."""

    blended_df: pd.DataFrame
    location_id: str
    fetch_timestamp: str
    weights_used: Dict[str, Dict[str, float]]  # variable -> {model: weight}
    available_models: List[str]
    missing_models: List[str]


class ForecastBlendEngine:
    """Engine for dynamically blending multi-model weather forecasts."""

    def __init__(
        self,
        default_weights: Optional[Dict[str, float]] = None,
        db_manager: Optional[Any] = None,
    ) -> None:
        self.default_weights = default_weights or DEFAULT_MODEL_WEIGHTS
        self.db = db_manager

    def _lookup_db_weights(
        self, location_id: str, season: str, regime: str = "all_regimes"
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """Query learned model weights from database for given location, season, and regime.

        Fallback chain: regime-specific → all_regimes → equal weighting.
        """
        if self.db is None:
            from ingestion.db import DatabaseManager
            self.db = DatabaseManager()

        df = self.db.get_model_weights(location_id=location_id, season=season, regime=regime)
        if df.empty and regime != "all_regimes":
            # Fallback: regime-specific not found → season-level aggregate
            df = self.db.get_model_weights(location_id=location_id, season=season, regime="all_regimes")
        if df.empty:
            # Try overall location weights if season-specific not present
            df = self.db.get_model_weights(location_id=location_id)

        if df.empty:
            return None

        learned_weights: Dict[str, Dict[str, float]] = {}
        for var, group in df.groupby("variable"):
            learned_weights[str(var)] = dict(zip(group["model"], group["weight"]))

        return learned_weights if learned_weights else None


    def blend(
        self,
        forecasts_df: pd.DataFrame,
        weights: Optional[Union[Dict[str, float], Dict[str, Dict[str, float]]]] = None,
        variables: Optional[List[str]] = None,
        obs_df: Optional[pd.DataFrame] = None,
    ) -> "BlendedForecastResult":
        """Blend forecasts across available models using weighted averaging.

        Args:
            forecasts_df: DataFrame containing normalized forecasts.
                Required columns: 'location_id', 'fetch_timestamp', 'target_time',
                                  'lead_time_hours', 'model', and forecast variables.
            weights: Optional custom weights. Can be either:
                - Dict[model_name, float] (applied across all variables)
                - Dict[variable_name, Dict[model_name, float]] (per-variable weights)
                If None, automatically queries learned weights from Module 2 (SQLite),
                conditioned on the detected weather regime, falling back to equal
                weighting if uncalibrated.
            variables: List of variable columns to blend. Defaults to BLEND_VARIABLES.
            obs_df: Optional recent observation DataFrame with columns 'time' and
                'precipitation' (used for regime classification).  If None, defaults
                to 'all_regimes' sentinel weights (backward-compatible behavior).

        Returns:
            BlendedForecastResult containing blended timeseries and weighting metadata.
        """
        if forecasts_df.empty:
            return BlendedForecastResult(
                blended_df=pd.DataFrame(),
                location_id="",
                fetch_timestamp="",
                weights_used={},
                available_models=[],
                missing_models=BLEND_MODELS,
            )

        variables = variables or BLEND_VARIABLES
        location_id = str(forecasts_df["location_id"].iloc[0])
        fetch_timestamp = str(forecasts_df["fetch_timestamp"].iloc[0])

        # If weights not explicitly provided, attempt lookup from learned weights table
        if weights is None:
            # Determine current season from first target time
            first_target = str(forecasts_df["target_time"].iloc[0])
            from weighting.skill import get_season_for_timestamp
            season = get_season_for_timestamp(first_target)

            # Detect weather regime from recent observations (live path)
            regime = "all_regimes"
            if obs_df is not None and not obs_df.empty:
                try:
                    from blending.regime import classify_regime
                    regime = classify_regime(obs_df, location_id, target_time=first_target)
                    logger.info(
                        "Regime detected for %s (%s): %s", location_id, season, regime
                    )
                except Exception as exc:
                    logger.warning(
                        "Regime classification failed for %s, using all_regimes: %s",
                        location_id, exc,
                    )

            learned_w = self._lookup_db_weights(
                location_id=location_id, season=season, regime=regime
            )
            if learned_w:
                weights = learned_w


        available_models = sorted(list(forecasts_df["model"].unique()))
        missing_models = [m for m in BLEND_MODELS if m not in available_models]

        if missing_models:
            logger.warning(
                "Blending with degraded/missing models for %s: missing %s; available %s",
                location_id,
                missing_models,
                available_models,
            )

        # Pivot forecasts by target_time and model
        target_times = sorted(forecasts_df["target_time"].unique())
        
        # Build lead_time mapping
        lead_time_map = (
            forecasts_df.groupby("target_time")["lead_time_hours"].min().to_dict()
        )

        weights_used: Dict[str, Dict[str, float]] = {}
        blended_rows: List[Dict[str, Any]] = []

        for t_str in target_times:
            sub = forecasts_df[forecasts_df["target_time"] == t_str]
            row_data: Dict[str, Any] = {
                "location_id": location_id,
                "fetch_timestamp": fetch_timestamp,
                "target_time": t_str,
                "lead_time_hours": lead_time_map.get(t_str, 0.0),
            }

            for var in variables:
                # Find available models that have non-null values for this timestamp and variable
                valid_sub = sub.dropna(subset=[var])
                models_with_data = valid_sub["model"].tolist()

                # Determine configured weights for this variable
                if isinstance(weights, dict) and var in weights and isinstance(weights[var], dict):
                    var_cfg_weights = weights[var]
                elif isinstance(weights, dict):
                    var_cfg_weights = weights  # flat dict
                else:
                    var_cfg_weights = self.default_weights

                # Renormalize weights across models with actual data at this step
                renorm_w = renormalize_weights(models_with_data, var_cfg_weights)
                weights_used[var] = renorm_w

                if not renorm_w:
                    row_data[f"{var}_blend"] = np.nan
                    row_data[f"{var}_min"] = np.nan
                    row_data[f"{var}_max"] = np.nan
                    continue

                # Compute weighted average
                vals = [
                    float(valid_sub[valid_sub["model"] == m][var].iloc[0])
                    for m in renorm_w.keys()
                ]
                w_vals = [renorm_w[m] for m in renorm_w.keys()]
                blended_val = float(np.average(vals, weights=w_vals))

                # Physical consistency enforcement (e.g. precipitation >= 0)
                if var == "precipitation":
                    blended_val = max(0.0, blended_val)

                row_data[f"{var}_blend"] = round(blended_val, 2)
                row_data[f"{var}_min"] = round(min(vals), 2)
                row_data[f"{var}_max"] = round(max(vals), 2)

                # Store individual model values in row for direct side-by-side comparison
                for m in BLEND_MODELS:
                    m_sub = valid_sub[valid_sub["model"] == m]
                    row_data[f"{var}_{m}"] = (
                        round(float(m_sub[var].iloc[0]), 2)
                        if not m_sub.empty
                        else None
                    )

            blended_rows.append(row_data)

        blended_df = pd.DataFrame(blended_rows)

        return BlendedForecastResult(
            blended_df=blended_df,
            location_id=location_id,
            fetch_timestamp=fetch_timestamp,
            weights_used=weights_used,
            available_models=available_models,
            missing_models=missing_models,
        )
