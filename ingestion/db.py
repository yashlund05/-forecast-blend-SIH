"""Database management for forecast blending system.

Implements SQLite relational tables for raw cache, normalized forecasts,
ensemble spread, historical observations, and pipeline execution logs.
"""

from contextlib import contextmanager
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, List, Optional
import pandas as pd

from ingestion.config import DEFAULT_DB_PATH

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite connections, migrations, and operations for forecast data."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Provide a transactional scope around database operations."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        """Create tables and indices if they do not exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Pipeline execution metadata table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    sources_attempted INTEGER NOT NULL,
                    sources_succeeded INTEGER NOT NULL,
                    details TEXT
                )
                """
            )

            # Raw API payload cache table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS raw_payload_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fetch_timestamp TEXT NOT NULL,
                    location_id TEXT NOT NULL,
                    endpoint_type TEXT NOT NULL,
                    payload_json TEXT,
                    is_success INTEGER NOT NULL,
                    error_message TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_raw_cache_lookup
                ON raw_payload_cache (location_id, endpoint_type, fetch_timestamp)
                """
            )

            # Normalized forecast table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS normalized_forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fetch_timestamp TEXT NOT NULL,
                    location_id TEXT NOT NULL,
                    target_time TEXT NOT NULL,
                    lead_time_hours REAL NOT NULL,
                    model TEXT NOT NULL,
                    temperature_2m REAL,
                    precipitation REAL,
                    wind_speed_10m REAL,
                    wind_gusts_10m REAL,
                    is_interpolated INTEGER DEFAULT 0
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_norm_lookup
                ON normalized_forecasts (location_id, target_time, model)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_norm_fetch
                ON normalized_forecasts (fetch_timestamp, location_id)
                """
            )

            # Ensemble spread and quantiles table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS ensemble_spread (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fetch_timestamp TEXT NOT NULL,
                    location_id TEXT NOT NULL,
                    target_time TEXT NOT NULL,
                    lead_time_hours REAL NOT NULL,
                    variable TEXT NOT NULL,
                    ensemble_mean REAL,
                    ensemble_std REAL,
                    ensemble_p10 REAL,
                    ensemble_p90 REAL
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ensemble_lookup
                ON ensemble_spread (location_id, target_time, variable)
                """
            )

            # Ground truth / historical observations table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS historical_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    location_id TEXT NOT NULL,
                    time TEXT NOT NULL,
                    temperature_2m REAL,
                    precipitation REAL,
                    wind_speed_10m REAL,
                    wind_gusts_10m REAL,
                    UNIQUE(location_id, time) ON CONFLICT REPLACE
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_obs_lookup
                ON historical_observations (location_id, time)
                """
            )

            # Historical model forecasts for backtesting and skill evaluation
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS historical_forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    location_id TEXT NOT NULL,
                    target_time TEXT NOT NULL,
                    model TEXT NOT NULL,
                    temperature_2m REAL,
                    precipitation REAL,
                    wind_speed_10m REAL,
                    UNIQUE(location_id, target_time, model) ON CONFLICT REPLACE
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_hist_fcst_lookup
                ON historical_forecasts (location_id, target_time, model)
                """
            )

            # Model weights lookup table (Module 2 output)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS model_weights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    location_id TEXT NOT NULL,
                    season TEXT NOT NULL,
                    lead_time_bucket TEXT NOT NULL,
                    variable TEXT NOT NULL,
                    model TEXT NOT NULL,
                    weight REAL NOT NULL,
                    sample_count INTEGER NOT NULL,
                    rmse REAL,
                    mae REAL,
                    low_confidence INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(location_id, season, lead_time_bucket, variable, model) ON CONFLICT REPLACE
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_model_weights_lookup
                ON model_weights (location_id, season, lead_time_bucket, variable)
                """
            )

    def save_raw_payload(
        self,
        fetch_timestamp: str,
        location_id: str,
        endpoint_type: str,
        payload: Optional[Dict[str, Any]],
        is_success: bool,
        error_message: Optional[str] = None,
    ) -> None:
        """Store raw API response payload or failure details for auditability."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO raw_payload_cache 
                (fetch_timestamp, location_id, endpoint_type, payload_json, is_success, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    fetch_timestamp,
                    location_id,
                    endpoint_type,
                    json.dumps(payload) if payload is not None else None,
                    1 if is_success else 0,
                    error_message,
                ),
            )

    def save_normalized_forecasts(self, df_or_records: Any) -> int:
        """Insert normalized forecast records into SQLite."""
        if isinstance(df_or_records, pd.DataFrame):
            if df_or_records.empty:
                return 0
            records = df_or_records.to_dict(orient="records")
        else:
            records = df_or_records

        if not records:
            return 0

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO normalized_forecasts (
                    fetch_timestamp, location_id, target_time, lead_time_hours,
                    model, temperature_2m, precipitation, wind_speed_10m,
                    wind_gusts_10m, is_interpolated
                ) VALUES (
                    :fetch_timestamp, :location_id, :target_time, :lead_time_hours,
                    :model, :temperature_2m, :precipitation, :wind_speed_10m,
                    :wind_gusts_10m, :is_interpolated
                )
                """,
                records,
            )
            return len(records)

    def save_ensemble_spread(self, df_or_records: Any) -> int:
        """Insert ensemble spread records into SQLite."""
        if isinstance(df_or_records, pd.DataFrame):
            if df_or_records.empty:
                return 0
            records = df_or_records.to_dict(orient="records")
        else:
            records = df_or_records

        if not records:
            return 0

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO ensemble_spread (
                    fetch_timestamp, location_id, target_time, lead_time_hours,
                    variable, ensemble_mean, ensemble_std, ensemble_p10, ensemble_p90
                ) VALUES (
                    :fetch_timestamp, :location_id, :target_time, :lead_time_hours,
                    :variable, :ensemble_mean, :ensemble_std, :ensemble_p10, :ensemble_p90
                )
                """,
                records,
            )
            return len(records)

    def save_historical_observations(self, df_or_records: Any) -> int:
        """Insert or replace historical ground-truth observations."""
        if isinstance(df_or_records, pd.DataFrame):
            if df_or_records.empty:
                return 0
            records = df_or_records.to_dict(orient="records")
        else:
            records = df_or_records

        if not records:
            return 0

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO historical_observations (
                    location_id, time, temperature_2m, precipitation,
                    wind_speed_10m, wind_gusts_10m
                ) VALUES (
                    :location_id, :time, :temperature_2m, :precipitation,
                    :wind_speed_10m, :wind_gusts_10m
                )
                """,
                records,
            )
            return len(records)

    def record_pipeline_run(
        self,
        run_timestamp: str,
        status: str,
        attempted: int,
        succeeded: int,
        details: str = "",
    ) -> int:
        """Log pipeline execution status and summary stats."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO pipeline_runs (
                    run_timestamp, status, sources_attempted, sources_succeeded, details
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (run_timestamp, status, attempted, succeeded, details),
            )
            return cursor.lastrowid or 0

    def get_latest_forecasts(
        self, location_id: str, fetch_timestamp: Optional[str] = None
    ) -> pd.DataFrame:
        """Retrieve the latest forecast records for a location as a DataFrame."""
        with self.get_connection() as conn:
            if fetch_timestamp is None:
                # Find most recent fetch timestamp for this location
                res = conn.execute(
                    """
                    SELECT fetch_timestamp FROM normalized_forecasts
                    WHERE location_id = ?
                    ORDER BY id DESC LIMIT 1
                    """,
                    (location_id,),
                ).fetchone()
                if not res:
                    return pd.DataFrame()
                fetch_timestamp = res["fetch_timestamp"]

            query = """
                SELECT location_id, fetch_timestamp, target_time, lead_time_hours,
                       model, temperature_2m, precipitation, wind_speed_10m,
                       wind_gusts_10m, is_interpolated
                FROM normalized_forecasts
                WHERE location_id = ? AND fetch_timestamp = ?
                ORDER BY target_time, model
            """
            return pd.read_sql_query(
                query, conn, params=(location_id, fetch_timestamp)
            )

    def get_historical_observations(
        self,
        location_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Retrieve ground-truth observations for a location within an optional date range."""
        with self.get_connection() as conn:
            query = "SELECT * FROM historical_observations WHERE location_id = ?"
            params: List[Any] = [location_id]
            if start_date:
                query += " AND time >= ?"
                params.append(start_date)
            if end_date:
                query += " AND time <= ?"
                params.append(end_date)
            query += " ORDER BY time"
            return pd.read_sql_query(query, conn, params=params)

    def get_pipeline_history(self, limit: int = 10) -> pd.DataFrame:
        """Retrieve recent pipeline execution logs."""
        with self.get_connection() as conn:
            query = "SELECT * FROM pipeline_runs ORDER BY id DESC LIMIT ?"
            return pd.read_sql_query(query, conn, params=(limit,))

    def save_historical_forecasts(self, df_or_records: Any) -> int:
        """Insert or replace historical model forecast runs."""
        if isinstance(df_or_records, pd.DataFrame):
            if df_or_records.empty:
                return 0
            records = df_or_records.to_dict(orient="records")
        else:
            records = df_or_records

        if not records:
            return 0

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO historical_forecasts (
                    location_id, target_time, model, temperature_2m, precipitation, wind_speed_10m
                ) VALUES (
                    :location_id, :target_time, :model, :temperature_2m, :precipitation, :wind_speed_10m
                )
                """,
                records,
            )
            return len(records)

    def get_historical_forecasts(
        self,
        location_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        models: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Retrieve historical model forecasts for a location."""
        with self.get_connection() as conn:
            query = "SELECT * FROM historical_forecasts WHERE location_id = ?"
            params: List[Any] = [location_id]
            if start_date:
                query += " AND target_time >= ?"
                params.append(start_date)
            if end_date:
                query += " AND target_time <= ?"
                params.append(end_date)
            if models:
                placeholders = ",".join(["?"] * len(models))
                query += f" AND model IN ({placeholders})"
                params.extend(models)
            query += " ORDER BY target_time, model"
            return pd.read_sql_query(query, conn, params=params)

    def save_model_weights(self, df_or_records: Any) -> int:
        """Insert or replace computed model weights."""
        if isinstance(df_or_records, pd.DataFrame):
            if df_or_records.empty:
                return 0
            records = df_or_records.to_dict(orient="records")
        else:
            records = df_or_records

        if not records:
            return 0

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO model_weights (
                    location_id, season, lead_time_bucket, variable,
                    model, weight, sample_count, rmse, mae, low_confidence, updated_at
                ) VALUES (
                    :location_id, :season, :lead_time_bucket, :variable,
                    :model, :weight, :sample_count, :rmse, :mae, :low_confidence, :updated_at
                )
                """,
                records,
            )
            return len(records)

    def get_model_weights(
        self,
        location_id: Optional[str] = None,
        season: Optional[str] = None,
        lead_time_bucket: Optional[str] = None,
        variable: Optional[str] = None,
    ) -> pd.DataFrame:
        """Query computed model weights from SQLite."""
        with self.get_connection() as conn:
            query = "SELECT * FROM model_weights WHERE 1=1"
            params: List[Any] = []
            if location_id:
                query += " AND location_id = ?"
                params.append(location_id)
            if season:
                query += " AND season = ?"
                params.append(season)
            if lead_time_bucket:
                query += " AND lead_time_bucket = ?"
                params.append(lead_time_bucket)
            if variable:
                query += " AND variable = ?"
                params.append(variable)
            query += " ORDER BY location_id, season, variable, model"
            return pd.read_sql_query(query, conn, params=params)

