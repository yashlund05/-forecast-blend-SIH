"""Historical observations backfill utility.

Pulls real historical reanalysis data from Open-Meteo Archive API
for training and verification time-split backtesting (Modules 2, 4, 5).
"""

import logging
import sys
import time
from typing import List, Optional

from ingestion.config import TARGET_LOCATIONS
from ingestion.db import DatabaseManager
from ingestion.pipeline import IngestionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backfill")

# Default 12-month historical window
DEFAULT_START_DATE = "2023-09-01"
DEFAULT_END_DATE = "2024-08-31"


def backfill_historical_data(
    locations: Optional[List[str]] = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> None:
    """Backfill historical archive observations for specified or all target locations."""
    db = DatabaseManager()
    pipeline = IngestionPipeline(db_manager=db)

    target_keys = locations or list(TARGET_LOCATIONS.keys())
    logger.info(
        "Starting historical reanalysis backfill for %d locations (%s to %s)",
        len(target_keys),
        start_date,
        end_date,
    )

    success_count = 0
    total_records = 0

    for idx, loc_id in enumerate(target_keys, 1):
        loc_cfg = TARGET_LOCATIONS[loc_id]
        logger.info(
            "[%d/%d] Fetching archive for %s (%s)...",
            idx,
            len(target_keys),
            loc_cfg.name,
            loc_id,
        )
        try:
            res = pipeline.ingest_archive_for_location(
                location_id=loc_id,
                start_date=start_date,
                end_date=end_date,
            )
            if res["is_success"]:
                records = res["records_saved"]
                total_records += records
                success_count += 1
                logger.info("  -> Saved %d hourly observations for %s", records, loc_id)
            else:
                logger.error("  -> Failed to ingest %s: %s", loc_id, res.get("error"))
        except Exception as e:
            logger.error("  -> Exception ingesting %s: %s", loc_id, e)

        # Brief rate-limit courtesy pause
        time.sleep(0.5)

    logger.info(
        "Historical backfill complete! Succeeded: %d/%d locations, Total observations saved: %d",
        success_count,
        len(target_keys),
        total_records,
    )


if __name__ == "__main__":
    locs = sys.argv[1:] if len(sys.argv) > 1 else None
    backfill_historical_data(locations=locs)
