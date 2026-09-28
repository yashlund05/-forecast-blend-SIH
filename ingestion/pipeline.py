"""Ingestion pipeline coordinator.

Orchestrates multi-source fetching, caching, normalization, and database storage.
Enforces per-source error resilience (a failing source degrades, never crashes the pipeline).
"""

from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, List, Optional
import pandas as pd

from ingestion.clients import (
    AIFSClient,
    ArchiveClient,
    EnsembleClient,
    MultiModelClient,
    WeatherNextClient,
)
from ingestion.config import (
    DEFAULT_FORECAST_DAYS,
    LocationConfig,
    TARGET_LOCATIONS,
)
from ingestion.db import DatabaseManager
from ingestion.normalizer import DataNormalizer

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """End-to-end ingestion orchestrator for weather forecasts and archives."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        multimodel_client: Optional[MultiModelClient] = None,
        aifs_client: Optional[AIFSClient] = None,
        weathernext_client: Optional[WeatherNextClient] = None,
        ensemble_client: Optional[EnsembleClient] = None,
        archive_client: Optional[ArchiveClient] = None,
    ) -> None:
        self.db = db_manager or DatabaseManager()
        self.multimodel_client = multimodel_client or MultiModelClient()
        self.aifs_client = aifs_client or AIFSClient()
        self.weathernext_client = weathernext_client or WeatherNextClient()
        self.ensemble_client = ensemble_client or EnsembleClient()
        self.archive_client = archive_client or ArchiveClient()

    def run_live_forecast_ingestion(
        self,
        locations: Optional[List[str]] = None,
        forecast_days: int = DEFAULT_FORECAST_DAYS,
    ) -> Dict[str, Any]:
        """Fetch and store live forecasts across all configured sources for target locations.
        
        Degrades gracefully if one or more sources fail.
        """
        fetch_timestamp = datetime.now(timezone.utc).isoformat()
        target_keys = locations or list(TARGET_LOCATIONS.keys())

        total_sources_attempted = 0
        total_sources_succeeded = 0
        location_summaries: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "Starting live forecast ingestion for %d locations at %s",
            len(target_keys),
            fetch_timestamp,
        )

        for loc_id in target_keys:
            if loc_id not in TARGET_LOCATIONS:
                logger.warning("Location %s not recognized in config. Skipping.", loc_id)
                continue

            loc_cfg = TARGET_LOCATIONS[loc_id]
            loc_attempted = 0
            loc_succeeded = 0
            loc_errors: List[str] = []

            # 1. Fetch Multi-Model NWP (GFS, ICON, ECMWF IFS)
            loc_attempted += 1
            mm_res = self.multimodel_client.fetch(
                latitude=loc_cfg.latitude,
                longitude=loc_cfg.longitude,
                forecast_days=forecast_days,
            )
            self.db.save_raw_payload(
                fetch_timestamp=fetch_timestamp,
                location_id=loc_id,
                endpoint_type="nwp_multimodel",
                payload=mm_res.data,
                is_success=mm_res.is_success,
                error_message=mm_res.error_message,
            )
            if mm_res.is_success and mm_res.data:
                loc_succeeded += 1
                norm_mm = DataNormalizer.normalize_multimodel(
                    raw_data=mm_res.data,
                    location_id=loc_id,
                    fetch_timestamp=fetch_timestamp,
                )
                self.db.save_normalized_forecasts(norm_mm)
            else:
                loc_errors.append(f"Multi-Model: {mm_res.error_message}")

            # 2. Fetch ECMWF AIFS (AI ML model)
            loc_attempted += 1
            aifs_res = self.aifs_client.fetch(
                latitude=loc_cfg.latitude,
                longitude=loc_cfg.longitude,
                forecast_days=forecast_days,
            )
            self.db.save_raw_payload(
                fetch_timestamp=fetch_timestamp,
                location_id=loc_id,
                endpoint_type="ai_aifs",
                payload=aifs_res.data,
                is_success=aifs_res.is_success,
                error_message=aifs_res.error_message,
            )
            if aifs_res.is_success and aifs_res.data:
                loc_succeeded += 1
                norm_aifs = DataNormalizer.normalize_aifs(
                    raw_data=aifs_res.data,
                    location_id=loc_id,
                    fetch_timestamp=fetch_timestamp,
                )
                self.db.save_normalized_forecasts(norm_aifs)
            else:
                loc_errors.append(f"AIFS: {aifs_res.error_message}")

            # 3. Fetch Google DeepMind WeatherNext 2 (AI/ML global model)
            loc_attempted += 1
            wn_res = self.weathernext_client.fetch(
                latitude=loc_cfg.latitude,
                longitude=loc_cfg.longitude,
                forecast_days=forecast_days,
            )
            self.db.save_raw_payload(
                fetch_timestamp=fetch_timestamp,
                location_id=loc_id,
                endpoint_type="ai_weathernext",
                payload=wn_res.data,
                is_success=wn_res.is_success,
                error_message=wn_res.error_message,
            )
            if wn_res.is_success and wn_res.data:
                loc_succeeded += 1
                norm_wn = DataNormalizer.normalize_weathernext(
                    raw_data=wn_res.data,
                    location_id=loc_id,
                    fetch_timestamp=fetch_timestamp,
                )
                self.db.save_normalized_forecasts(norm_wn)
            else:
                loc_errors.append(f"WeatherNext: {wn_res.error_message}")

            # 4. Fetch Ensemble Spread
            loc_attempted += 1
            ens_res = self.ensemble_client.fetch(
                latitude=loc_cfg.latitude,
                longitude=loc_cfg.longitude,
                forecast_days=forecast_days,
            )
            self.db.save_raw_payload(
                fetch_timestamp=fetch_timestamp,
                location_id=loc_id,
                endpoint_type="ensemble",
                payload=ens_res.data,
                is_success=ens_res.is_success,
                error_message=ens_res.error_message,
            )
            if ens_res.is_success and ens_res.data:
                loc_succeeded += 1
                norm_ens = DataNormalizer.normalize_ensemble(
                    raw_data=ens_res.data,
                    location_id=loc_id,
                    fetch_timestamp=fetch_timestamp,
                )
                self.db.save_ensemble_spread(norm_ens)
            else:
                loc_errors.append(f"Ensemble: {ens_res.error_message}")

            total_sources_attempted += loc_attempted
            total_sources_succeeded += loc_succeeded

            location_summaries[loc_id] = {
                "attempted": loc_attempted,
                "succeeded": loc_succeeded,
                "status": "full" if loc_succeeded == loc_attempted else "degraded" if loc_succeeded > 0 else "failed",
                "errors": loc_errors,
            }

        # Overall pipeline run status
        if total_sources_succeeded == total_sources_attempted:
            overall_status = "success"
        elif total_sources_succeeded > 0:
            overall_status = "degraded"
        else:
            overall_status = "failed"

        details_json = json.dumps(
            {
                "timestamp": fetch_timestamp,
                "locations": location_summaries,
            }
        )

        run_id = self.db.record_pipeline_run(
            run_timestamp=fetch_timestamp,
            status=overall_status,
            attempted=total_sources_attempted,
            succeeded=total_sources_succeeded,
            details=details_json,
        )

        logger.info(
            "Ingestion run %d finished with status '%s' (%d/%d sources succeeded)",
            run_id,
            overall_status,
            total_sources_succeeded,
            total_sources_attempted,
        )

        return {
            "run_id": run_id,
            "timestamp": fetch_timestamp,
            "status": overall_status,
            "sources_attempted": total_sources_attempted,
            "sources_succeeded": total_sources_succeeded,
            "locations": location_summaries,
        }

    def ingest_archive_for_location(
        self,
        location_id: str,
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """Fetch and store historical reanalysis observations for ground-truth verification."""
        if location_id not in TARGET_LOCATIONS:
            raise ValueError(f"Unknown location_id: {location_id}")

        loc_cfg = TARGET_LOCATIONS[location_id]
        fetch_timestamp = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Fetching historical archive for %s from %s to %s",
            location_id,
            start_date,
            end_date,
        )

        result = self.archive_client.fetch(
            latitude=loc_cfg.latitude,
            longitude=loc_cfg.longitude,
            start_date=start_date,
            end_date=end_date,
        )

        self.db.save_raw_payload(
            fetch_timestamp=fetch_timestamp,
            location_id=location_id,
            endpoint_type="historical_archive",
            payload=result.data,
            is_success=result.is_success,
            error_message=result.error_message,
        )

        if not result.is_success or not result.data:
            return {
                "location_id": location_id,
                "is_success": False,
                "error": result.error_message,
                "records_saved": 0,
            }

        norm_obs = DataNormalizer.normalize_archive(
            raw_data=result.data,
            location_id=location_id,
        )
        saved_count = self.db.save_historical_observations(norm_obs)

        return {
            "location_id": location_id,
            "is_success": True,
            "records_saved": saved_count,
        }
