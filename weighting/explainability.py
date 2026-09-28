"""Model Weight Explainability Engine (Module 2 & Phase 7).

Provides transparent, audit-ready "Why This Weight?" mathematical traces
and physical meteorological rationale for any station, season, and variable.
Demonstrates the exact derivation from historical reanalysis RMSE to final blend weights.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd

from ingestion.config import TARGET_LOCATIONS
from ingestion.db import DatabaseManager
from weighting.skill import INVERSE_ERROR_POWER

EPSILON: float = 1e-4
INVERSE_POWER: float = INVERSE_ERROR_POWER

# Physical meteorological explanations by topography and variable
TOPOGRAPHY_RATIONALE: Dict[str, Dict[str, str]] = {
    "coastal": {
        "precipitation": (
            "Maritime boundary layer dynamics and onshore monsoonal moisture flux are resolved with higher accuracy "
            "by ECMWF IFS (higher vertical resolution and advanced 4D-Var data assimilation of satellite radiances), "
            "yielding a significantly lower historical RMSE compared to global convective parameterizations in GFS/ICON."
        ),
        "temperature_2m": (
            "Coastal land-sea breeze circulations moderate diurnal temperature spikes. ECMWF IFS and ICON capture "
            "coastal surface roughness and maritime humidity accurately, stabilizing temperature predictions."
        ),
        "wind_speed_10m": (
            "Coastal and maritime surface friction variations favor ECMWF IFS and GFS, which incorporate offshore scatterometer "
            "wind observations during synoptic depressions."
        ),
    },
    "plains": {
        "precipitation": (
            "Continental monsoon trough dynamics and localized mesoscale convective complexes produce varied skill. "
            "ECMWF IFS leads on synoptic rain bands, while GFS provides strong complementary skill during dry spells."
        ),
        "temperature_2m": (
            "In semi-arid and alluvial plains, strong radiative surface heating and diurnal boundary layer growth "
            "are well simulated by NOAA GFS and ECMWF IFS, producing balanced inverse-error weighting."
        ),
        "wind_speed_10m": (
            "Flat topography minimizes orographic turbulence; all three NWP models exhibit low error, resulting in "
            "closely distributed weights."
        ),
    },
    "arid": {
        "precipitation": (
            "Sporadic desert rain events are sparse in Thar. Models with lower false-alarm ratios receive higher trust."
        ),
        "temperature_2m": (
            "High sensible heat flux and extreme diurnal amplitude in the desert require robust soil-moisture physics. "
            "GFS and ECMWF IFS demonstrate consistent skill, with GFS capturing desert radiative cooling accurately."
        ),
        "wind_speed_10m": (
            "Thermal lows generate gusty afternoon surface winds. DWD ICON and GFS show strong surface layer skill."
        ),
    },
    "hill": {
        "precipitation": (
            "Complex Himalayan terrain induces strong orographic lift and rain-shadow effects. Global models (0.25° grid) "
            "smooth mountain crests, leading to elevation biases. Inverse-error weighting penalizes large orographic busts "
            "and distributes weights across models to prevent single-model mountain failure."
        ),
        "temperature_2m": (
            "Lapse-rate adjustments and valley cold-air pooling challenge coarse terrain grids. ECMWF IFS demonstrates "
            "the most consistent lapse-rate fidelity among global sources."
        ),
        "wind_speed_10m": (
            "Valley channelling and mountain-wave turbulence create localized gustiness that requires distributed multi-model blending."
        ),
    },
    "deltaic": {
        "precipitation": (
            "Monsoon depressions from the Bay of Bengal make landfall over deltaic floodplains. ECMWF IFS's moist physics "
            "consistently yields the lowest RMSE during heavy cyclonic rain episodes."
        ),
        "temperature_2m": (
            "High ambient relative humidity dampens diurnal temperature range; ECMWF IFS and ICON maintain close tracking."
        ),
        "wind_speed_10m": (
            "Cyclone-exposed delta requires high sensitivity to coastal gale transitions; ECMWF IFS is prioritized based on empirical verification."
        ),
    },
}


@dataclass
class ExplainabilityTrace:
    """Audit-ready explainability trace explaining weight derivations."""

    location_id: str
    station_name: str
    zone: str
    topography: str
    season: str
    variable: str
    lead_time_bucket: str
    formula: str
    inverse_power_p: float
    epsilon: float
    models: List[Dict[str, Any]]
    meteorological_rationale: str


class WeightExplainabilityEngine:
    """Generates transparent step-by-step mathematical explanations of learned blend weights."""

    def __init__(self, db: Optional[DatabaseManager] = None) -> None:
        self.db = db or DatabaseManager()

    def explain_weights(
        self,
        location_id: str,
        season: str = "monsoon",
        variable: str = "precipitation",
        lead_time_bucket: str = "all",
    ) -> ExplainabilityTrace:
        """Compute the mathematical derivation and physical rationale for a given context."""
        loc_cfg = TARGET_LOCATIONS.get(location_id)
        if not loc_cfg:
            raise ValueError(f"Unknown location: {location_id}")

        weights_df = self.db.get_model_weights(
            location_id=location_id,
            season=season,
            variable=variable,
            lead_time_bucket=lead_time_bucket,
        )

        model_records: List[Dict[str, Any]] = []
        formula_str = f"W_m = (1 / (RMSE_m^{INVERSE_POWER} + {EPSILON})) / SUM(1 / (RMSE_k^{INVERSE_POWER} + {EPSILON}))"

        if not weights_df.empty:
            for _, row in weights_df.iterrows():
                m = str(row["model"])
                w = float(row["weight"])
                rmse = float(row["rmse"]) if pd.notnull(row["rmse"]) else None
                mae = float(row["mae"]) if pd.notnull(row["mae"]) else None
                n_samples = int(row["sample_count"])
                low_conf = bool(row["low_confidence"])

                if rmse is not None:
                    inv_err_score = 1.0 / ((rmse ** INVERSE_POWER) + EPSILON)
                else:
                    inv_err_score = None

                status_label = "Uncalibrated Baseline (Equal Share)" if low_conf else f"Calibrated ({n_samples:,} reanalysis samples)"

                model_records.append(
                    {
                        "model": m,
                        "display_name": {
                            "ecmwf_ifs": "ECMWF IFS (0.25° Physics)",
                            "ecmwf_aifs": "ECMWF AIFS (0.25° AI/ML)",
                            "weathernext": "Google WeatherNext 2 (0.25° AI/ML)",
                            "gfs": "NOAA GFS (0.25° Physics)",
                            "icon": "DWD ICON (0.25° Physics)",
                        }.get(m, m),
                        "rmse": rmse,
                        "mae": mae,
                        "sample_count": n_samples,
                        "inv_score": round(inv_err_score, 4) if inv_err_score else None,
                        "final_weight": round(w, 4),
                        "final_weight_pct": f"{w * 100:.1f}%",
                        "status": status_label,
                    }
                )

        # Get meteorological rationale
        topo_dict = TOPOGRAPHY_RATIONALE.get(loc_cfg.topography.lower(), TOPOGRAPHY_RATIONALE["plains"])
        phys_rationale = topo_dict.get(
            variable,
            f"Weights are determined strictly by empirical RMSE minimization across {loc_cfg.topography} terrain."
        )

        return ExplainabilityTrace(
            location_id=location_id,
            station_name=loc_cfg.name,
            zone=loc_cfg.zone,
            topography=loc_cfg.topography,
            season=season,
            variable=variable,
            lead_time_bucket=lead_time_bucket,
            formula=formula_str,
            inverse_power_p=INVERSE_POWER,
            epsilon=EPSILON,
            models=model_records,
            meteorological_rationale=phys_rationale,
        )
