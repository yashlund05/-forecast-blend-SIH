"""Model Weight Generator and Lookup Engine (Module 2).

Converts empirical RMSE metrics into inverse-error weights (w_m ∝ 1 / RMSE_m^2).
Handles sample threshold enforcement, fallback to equal weighting on low confidence,
and SQLite persistence.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
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

    # If some models are missing samples (e.g. newly introduced AI models AIFS / WeatherNext 2),
    # allocate a baseline equal share to uncalibrated models and distribute the rest
    # among calibrated models according to inverse error, marking low_confidence=True.
    weights: Dict[str, float] = {}
    if len(valid_inv_scores) < n_models:
        # Each uncalibrated model gets an equal baseline slice (1 / n_models)
        uncalibrated_per_model = equal_weight
        num_uncalibrated = n_models - len(valid_inv_scores)
        uncalibrated_share = uncalibrated_per_model * num_uncalibrated
        calibrated_share = max(0.0, 1.0 - uncalibrated_share)

        for m in models:
            if m in valid_inv_scores:
                weights[m] = round(
                    calibrated_share * (valid_inv_scores[m] / total_inv), 4
                )
            else:
                weights[m] = round(uncalibrated_per_model, 4)
        is_low_confidence = len(valid_inv_scores) < 2
    else:
        for m in models:
            weights[m] = round(valid_inv_scores[m] / total_inv, 4)
        is_low_confidence = False

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
        """Compute skill metrics and generate full weights table stored in SQLite.

        The weight table now conditions on four dimensions:
          location × season × regime × lead_time_bucket × variable
        where regime ∈ {monsoon_active, monsoon_break, off_season, all_regimes}.
        'all_regimes' is an aggregate sentinel retained for backward-compatible lookups.
        """
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
            lead_buckets = skill_df["lead_time_bucket"].unique() if "lead_time_bucket" in skill_df.columns else ["all"]
            regimes = skill_df["regime"].unique() if "regime" in skill_df.columns else ["all_regimes"]

            for season in seasons:
                for regime in regimes:
                    for bucket in lead_buckets:
                        for var in variables:
                            sub = skill_df[
                                (skill_df["season"] == season)
                                & (skill_df["regime"] == regime)
                                & (skill_df["lead_time_bucket"] == bucket)
                                & (skill_df["variable"] == var)
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
                                        "regime": regime,
                                        "lead_time_bucket": bucket,
                                        "variable": var,
                                        "model": model,
                                        "weight": w,
                                        "sample_count": sample_map.get(model, 0),
                                        "rmse": rmse_map.get(model),
                                        "mae": mae_map.get(model),
                                        "low_confidence": 1 if (sample_map.get(model, 0) < MIN_SAMPLE_THRESHOLD or is_low_conf) else 0,
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
        regime: str = "all_regimes",
    ) -> Dict[str, Dict[str, float]]:
        """Retrieve weights dictionary formatted as Dict[variable, Dict[model, weight]].

        Fallback chain:
          1. regime-specific weights (e.g. 'monsoon_active')
          2. 'all_regimes' aggregate (if regime-specific is empty)
          3. equal weighting among BLEND_MODELS (if nothing in DB)

        Directly compatible with ForecastBlendEngine.blend(weights=...).
        """
        df = self.db.get_model_weights(
            location_id=location_id,
            season=season,
            regime=regime,
            lead_time_bucket=lead_time_bucket,
        )

        # Fallback 1: regime not found → try all_regimes sentinel
        if df.empty and regime != "all_regimes":
            df = self.db.get_model_weights(
                location_id=location_id,
                season=season,
                regime="all_regimes",
                lead_time_bucket=lead_time_bucket,
            )

        # Fallback 2: lead-time bucket not found → try "all" aggregate
        if df.empty and lead_time_bucket != "all":
            df = self.db.get_model_weights(
                location_id=location_id,
                season=season,
                regime="all_regimes",
                lead_time_bucket="all",
            )

        if df.empty:
            # Fallback 3: equal weighting
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
        regime: str = "all_regimes",
    ) -> pd.DataFrame:
        """Return the highest-weighted model per station for map visualization.

        Args:
            season: Season name.
            lead_time_bucket: Lead-time bucket (defaults to 'all').
            variable: Forecast variable.
            regime: Weather regime ('all_regimes', 'monsoon_active', 'monsoon_break',
                    'off_season').  Defaults to 'all_regimes' aggregate.
        """
        df = self.db.get_model_weights(
            season=season,
            regime=regime,
            lead_time_bucket=lead_time_bucket,
            variable=variable,
        )
        if df.empty and lead_time_bucket != "all":
            df = self.db.get_model_weights(
                season=season,
                regime=regime,
                lead_time_bucket="all",
                variable=variable,
            )
        if df.empty and regime != "all_regimes":
            df = self.db.get_model_weights(
                season=season,
                regime="all_regimes",
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
                        "weight": round(1.0 / len(BLEND_MODELS), 4),
                        "rmse": None,
                        "low_confidence": True,
                    }
                )

        return pd.DataFrame(rows)


def interpolate_weight_grid(
    station_lats: np.ndarray,
    station_lons: np.ndarray,
    station_values: np.ndarray,
    grid_lat_min: float = 8.0,
    grid_lat_max: float = 36.0,
    grid_lon_min: float = 68.0,
    grid_lon_max: float = 96.0,
    n_points_lat: int = 25,
    n_points_lon: int = 25,
    power: float = 2.0,
    eps: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coarse spatial interpolation of station weights onto a regular national grid using IDW.

    CAVEAT / METHODOLOGICAL DISCLAIMER:
    This spatial interpolation is a coarse visual representation derived from only 10 national
    stations across India. It does NOT claim true gridded NWP skill or fine-scale local variance.
    Its purpose is to provide forecasters and technical evaluators with an indicative regional
    contour of model dominance and relative weight transitions across synoptic zones.

    Args:
        station_lats: Array of station latitudes (1D).
        station_lons: Array of station longitudes (1D).
        station_values: Array of station values/weights (1D).
        grid_lat_min: Southern boundary (default 8°N).
        grid_lat_max: Northern boundary (default 36°N).
        grid_lon_min: Western boundary (default 68°E).
        grid_lon_max: Eastern boundary (default 96°E).
        n_points_lat: Number of latitude grid divisions.
        n_points_lon: Number of longitude grid divisions.
        power: Distance decay exponent (default 2.0 for quadratic inverse distance).
        eps: Small tolerance to avoid division by zero at station coordinates.

    Returns:
        (grid_lats, grid_lons, interpolated_grid_2d)
        - grid_lats: 1D array of latitude coordinates.
        - grid_lons: 1D array of longitude coordinates.
        - interpolated_grid_2d: 2D array of interpolated values (shape: [n_points_lat, n_points_lon]).
    """
    lats = np.asarray(station_lats, dtype=float)
    lons = np.asarray(station_lons, dtype=float)
    vals = np.asarray(station_values, dtype=float)

    if len(lats) == 0 or len(vals) == 0:
        raise ValueError("Cannot interpolate with empty station inputs.")

    grid_lats = np.linspace(grid_lat_min, grid_lat_max, n_points_lat)
    grid_lons = np.linspace(grid_lon_min, grid_lon_max, n_points_lon)

    # Meshgrid: shape (n_points_lat, n_points_lon)
    mesh_lons, mesh_lats = np.meshgrid(grid_lons, grid_lats)

    # Euclidean distance in lat-lon space: shape (n_lat, n_lon, n_stations)
    dists = np.sqrt(
        (mesh_lats[:, :, None] - lats[None, None, :]) ** 2 +
        (mesh_lons[:, :, None] - lons[None, None, :]) ** 2
    )

    exact_matches = dists < eps
    with np.errstate(divide="ignore"):
        idw_weights = 1.0 / (dists ** power)

    # Assign high finite weight for exact station coordinates
    idw_weights = np.where(exact_matches, 1e12, idw_weights)
    total_weights = np.sum(idw_weights, axis=-1, keepdims=True)

    # Compute weighted sum
    interp_grid = np.sum(idw_weights * vals[None, None, :], axis=-1) / np.squeeze(total_weights, axis=-1)

    return grid_lats, grid_lons, interp_grid


# Named constant for geographic cutoff
# Stations spaced across India have regional representation up to ~500 km.
# Beyond 500 km, IDW extrapolation is ungrounded even within Indian borders,
# so cells exceeding this threshold are masked out to prevent misleading claims.
MAX_INTERPOLATION_DISTANCE_KM: float = 500.0


def _point_in_multipolygon(
    lons: np.ndarray,
    lats: np.ndarray,
    polygons: List[List[List[float]]],
) -> np.ndarray:
    """Vectorized ray-casting algorithm to test whether (lon, lat) points lie inside a MultiPolygon.

    Args:
        lons: 1D array of longitude coordinates.
        lats: 1D array of latitude coordinates.
        polygons: List of polygon rings, each ring being a list of [lon, lat] coordinates.

    Returns:
        Boolean 1D array indicating whether each point is inside any polygon ring.
    """
    inside = np.zeros(len(lons), dtype=bool)
    x = lons
    y = lats

    for poly_coords in polygons:
        poly_arr = np.asarray(poly_coords, dtype=float)
        n = len(poly_arr)
        if n < 3:
            continue
        poly_inside = np.zeros(len(x), dtype=bool)
        p1x, p1y = poly_arr[0]
        for i in range(1, n + 1):
            p2x, p2y = poly_arr[i % n]
            # Edge crossing condition
            mask = (y > min(p1y, p2y)) & (y <= max(p1y, p2y)) & (x <= max(p1x, p2x))
            if np.any(mask):
                xinters = (y[mask] - p1y) * (p2x - p1x) / (p2y - p1y + 1e-12) + p1x
                cross = x[mask] <= xinters
                poly_inside[mask] = poly_inside[mask] ^ cross
            p1x, p1y = p2x, p2y
        inside = inside | poly_inside

    return inside


def _haversine_distance_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: float,
    lon2: float,
) -> np.ndarray:
    """Computes great-circle distances in kilometers between an array of points and a reference coordinate."""
    R = 6371.0  # Earth mean radius in kilometers
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * (np.sin(delta_lambda / 2.0) ** 2)
    )
    return 2.0 * R * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))


def mask_grid_to_india(
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    interp_grid: np.ndarray,
    station_lats: np.ndarray,
    station_lons: np.ndarray,
    geojson_path: Optional[str] = None,
    max_distance_km: float = MAX_INTERPOLATION_DISTANCE_KM,
) -> np.ndarray:
    """Masks IDW grid cells that lie outside India's sovereign boundaries or beyond station influence.

    Scope & Design Rationale:
    1. Border Clipping: An unmasked bounding box (8°-36°N, 68°-96°E) paints skill over the
       Arabian Sea, Bay of Bengal, Pakistan, Nepal, and China. Using the verified Natural Earth
       administrative boundary (committed as data/india_boundary.geojson) masks foreign territories
       and ocean cells without adding heavy GIS binary dependencies.
    2. Station Proximity Filter: Even within India, cells farther than max_distance_km
       (default 500 km) from any observation/forecast station have no physical or empirical
       justification for IDW weighting and are masked with NaN.

    Args:
        grid_lats: 1D array of latitude coordinates from interpolate_weight_grid.
        grid_lons: 1D array of longitude coordinates from interpolate_weight_grid.
        interp_grid: 2D array of interpolated values (shape: [n_points_lat, n_points_lon]).
        station_lats: 1D array of station latitude coordinates.
        station_lons: 1D array of station longitude coordinates.
        geojson_path: Optional path to GeoJSON file. Defaults to data/india_boundary.geojson.
        max_distance_km: Maximum allowable distance from the nearest station in km.

    Returns:
        2D masked array of same shape as interp_grid, with invalid cells set to np.nan.
    """
    if geojson_path is None:
        geojson_path = str(Path(__file__).resolve().parent.parent / "data" / "india_boundary.geojson")

    path_obj = Path(geojson_path)
    if not path_obj.exists():
        logger.warning(f"GeoJSON boundary file {geojson_path} not found; returning unmasked grid.")
        return interp_grid.copy()

    with open(path_obj, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract all coordinate polygon rings from FeatureCollection or Geometry
    polygons: List[List[List[float]]] = []
    features = data.get("features", [])
    for feat in features:
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        if gtype == "Polygon":
            if coords:
                polygons.append(coords[0])
        elif gtype == "MultiPolygon":
            for poly in coords:
                if poly:
                    polygons.append(poly[0])

    mesh_lons, mesh_lats = np.meshgrid(grid_lons, grid_lats)
    flat_lats = mesh_lats.flatten()
    flat_lons = mesh_lons.flatten()

    # 1. Point-in-polygon check for India boundary
    inside_india = _point_in_multipolygon(flat_lons, flat_lats, polygons)

    # 2. Distance check to nearest station
    stn_lats = np.asarray(station_lats, dtype=float)
    stn_lons = np.asarray(station_lons, dtype=float)

    if len(stn_lats) > 0:
        dists_km = np.empty((len(flat_lats), len(stn_lats)), dtype=float)
        for j in range(len(stn_lats)):
            dists_km[:, j] = _haversine_distance_km(flat_lats, flat_lons, stn_lats[j], stn_lons[j])
        min_dist_to_station = np.min(dists_km, axis=1)
        valid_distance = min_dist_to_station <= max_distance_km
    else:
        valid_distance = np.ones(len(flat_lats), dtype=bool)

    # Valid mask combines sovereign territory and station proximity
    valid_cells = inside_india & valid_distance

    masked_flat = interp_grid.flatten().copy()
    masked_flat[~valid_cells] = np.nan

    return masked_flat.reshape(interp_grid.shape)



