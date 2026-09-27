"""Data normalizer for converting diverse Open-Meteo API outputs into unified schemas.

Reconciles model naming, resolves temporal resolution mismatches (e.g. AIFS timesteps),
and computes ensemble spread statistics (mean, std, p10, p90).
"""

from datetime import datetime
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from ingestion.config import MODEL_MAP

logger = logging.getLogger(__name__)


def parse_iso_datetime(dt_str: str) -> datetime:
    """Safely parse Open-Meteo ISO 8601 timestamp string."""
    clean_str = dt_str.replace("Z", "+00:00")
    if "+" not in clean_str and "-" not in clean_str[10:]:
        # Default to UTC if timezone offset is not explicitly included
        clean_str += "+00:00"
    return datetime.fromisoformat(clean_str)


class DataNormalizer:
    """Normalizes raw API responses from multiple forecast models and archives."""

    @staticmethod
    def normalize_multimodel(
        raw_data: Dict[str, Any],
        location_id: str,
        fetch_timestamp: str,
    ) -> pd.DataFrame:
        """Normalize physical NWP multi-model forecast into standardized records."""
        if not raw_data or "hourly" not in raw_data:
            return pd.DataFrame()

        hourly = raw_data["hourly"]
        times = hourly.get("time", [])
        if not times:
            return pd.DataFrame()

        fetch_dt = parse_iso_datetime(fetch_timestamp)

        records: List[Dict[str, Any]] = []

        models_to_extract = [
            ("gfs_seamless", "gfs"),
            ("icon_seamless", "icon"),
            ("ecmwf_ifs025", "ecmwf_ifs"),
        ]

        for idx, t_str in enumerate(times):
            target_dt = parse_iso_datetime(t_str)
            lead_time_hours = max(
                0.0, (target_dt - fetch_dt).total_seconds() / 3600.0
            )

            for raw_model, canon_model in models_to_extract:
                temp_key = f"temperature_2m_{raw_model}"
                precip_key = f"precipitation_{raw_model}"
                wind_key = f"wind_speed_10m_{raw_model}"
                gust_key = f"wind_gusts_10m_{raw_model}"

                # Only include model if present in hourly payload
                if temp_key in hourly:
                    temp_val = hourly[temp_key][idx]
                    precip_val = hourly.get(precip_key, [None])[idx]
                    wind_val = hourly.get(wind_key, [None])[idx]
                    gust_val = hourly.get(gust_key, [None])[idx]

                    # Filter out entries where primary variables are null
                    if temp_val is not None:
                        records.append(
                            {
                                "fetch_timestamp": fetch_timestamp,
                                "location_id": location_id,
                                "target_time": t_str,
                                "lead_time_hours": round(lead_time_hours, 2),
                                "model": canon_model,
                                "temperature_2m": float(temp_val),
                                "precipitation": float(precip_val)
                                if precip_val is not None
                                else 0.0,
                                "wind_speed_10m": float(wind_val)
                                if wind_val is not None
                                else 0.0,
                                "wind_gusts_10m": float(gust_val)
                                if gust_val is not None
                                else None,
                                "is_interpolated": 0,
                            }
                        )

        return pd.DataFrame(records)

    @staticmethod
    def normalize_aifs(
        raw_data: Dict[str, Any],
        location_id: str,
        fetch_timestamp: str,
    ) -> pd.DataFrame:
        """Normalize ECMWF AIFS ML forecast.
        
        Handles potential 6-hourly step resolution by verifying step consistency,
        interpolating intervening hours where required, and documenting resolution alignment.
        """
        if not raw_data or "hourly" not in raw_data:
            return pd.DataFrame()

        hourly = raw_data["hourly"]
        times = hourly.get("time", [])
        if not times:
            return pd.DataFrame()

        fetch_dt = parse_iso_datetime(fetch_timestamp)
        temps = hourly.get("temperature_2m", [])
        precips = hourly.get("precipitation", [])
        winds = hourly.get("wind_speed_10m", [])

        # Create temporary series to inspect null distribution
        df = pd.DataFrame(
            {
                "target_time": times,
                "temperature_2m": temps,
                "precipitation": precips,
                "wind_speed_10m": winds,
            }
        )

        # Convert to numeric, leaving nulls intact
        for col in ["temperature_2m", "precipitation", "wind_speed_10m"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # If data is sparse (e.g. 6-hourly with intervening NaNs), interpolate explicitly
        valid_count = df["temperature_2m"].count()
        total_count = len(df)
        is_sparse = 0 < valid_count < (total_count * 0.8)

        if is_sparse:
            logger.info(
                "AIFS data for %s has %d/%d valid points. Interpolating step resolution.",
                location_id,
                valid_count,
                total_count,
            )
            df["temperature_2m"] = df["temperature_2m"].interpolate(method="linear")
            df["wind_speed_10m"] = df["wind_speed_10m"].interpolate(method="linear")
            # Precipitation accumulates; forward-fill / distribute across step
            df["precipitation"] = df["precipitation"].fillna(0.0)
            interpolated_flag = 1
        else:
            interpolated_flag = 0

        # Drop rows that remain null (e.g. before forecast start)
        df = df.dropna(subset=["temperature_2m"]).copy()
        if df.empty:
            return pd.DataFrame()

        df["fetch_timestamp"] = fetch_timestamp
        df["location_id"] = location_id
        df["model"] = "ecmwf_aifs"
        df["wind_gusts_10m"] = None  # AIFS doesn't natively predict surface gusts
        df["is_interpolated"] = interpolated_flag

        target_dts = [parse_iso_datetime(t) for t in df["target_time"]]
        df["lead_time_hours"] = [
            round(max(0.0, (tdt - fetch_dt).total_seconds() / 3600.0), 2)
            for tdt in target_dts
        ]

        # Reorder to standard schema
        cols = [
            "fetch_timestamp",
            "location_id",
            "target_time",
            "lead_time_hours",
            "model",
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_gusts_10m",
            "is_interpolated",
        ]
        return df[cols]

    @staticmethod
    def normalize_ensemble(
        raw_data: Dict[str, Any],
        location_id: str,
        fetch_timestamp: str,
    ) -> pd.DataFrame:
        """Compute ensemble distribution statistics (mean, std, p10, p90) per timestamp."""
        if not raw_data or "hourly" not in raw_data:
            return pd.DataFrame()

        hourly = raw_data["hourly"]
        times = hourly.get("time", [])
        if not times:
            return pd.DataFrame()

        fetch_dt = parse_iso_datetime(fetch_timestamp)

        variables = ["temperature_2m", "precipitation", "wind_speed_10m"]
        records: List[Dict[str, Any]] = []

        for var in variables:
            # Find all member columns for this variable
            member_cols = [
                col
                for col in hourly.keys()
                if col.startswith(f"{var}_member") or col == var
            ]
            if not member_cols:
                continue

            matrix = np.array([hourly[col] for col in member_cols], dtype=float)

            for idx, t_str in enumerate(times):
                target_dt = parse_iso_datetime(t_str)
                lead_time_hours = max(
                    0.0, (target_dt - fetch_dt).total_seconds() / 3600.0
                )

                vals = matrix[:, idx]
                vals = vals[~np.isnan(vals)]
                if len(vals) == 0:
                    continue

                records.append(
                    {
                        "fetch_timestamp": fetch_timestamp,
                        "location_id": location_id,
                        "target_time": t_str,
                        "lead_time_hours": round(lead_time_hours, 2),
                        "variable": var,
                        "ensemble_mean": float(np.mean(vals)),
                        "ensemble_std": float(np.std(vals)),
                        "ensemble_p10": float(np.percentile(vals, 10)),
                        "ensemble_p90": float(np.percentile(vals, 90)),
                    }
                )

        return pd.DataFrame(records)

    @staticmethod
    def normalize_archive(
        raw_data: Dict[str, Any],
        location_id: str,
    ) -> pd.DataFrame:
        """Normalize historical/archive reanalysis data into ground-truth observations."""
        if not raw_data or "hourly" not in raw_data:
            return pd.DataFrame()

        hourly = raw_data["hourly"]
        times = hourly.get("time", [])
        if not times:
            return pd.DataFrame()

        df = pd.DataFrame(
            {
                "location_id": location_id,
                "time": times,
                "temperature_2m": hourly.get("temperature_2m", [None] * len(times)),
                "precipitation": hourly.get("precipitation", [None] * len(times)),
                "wind_speed_10m": hourly.get("wind_speed_10m", [None] * len(times)),
                "wind_gusts_10m": hourly.get("wind_gusts_10m", [None] * len(times)),
            }
        )

        for col in ["temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["temperature_2m"]).copy()
        return df
