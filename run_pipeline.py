"""Operational entry point for forecast ingestion pipeline.

Supports both one-off runs and automated scheduling via the schedule library.
"""

import argparse
import logging
import sys
import time
from typing import List, Optional

import schedule

from ingestion.config import TARGET_LOCATIONS
from ingestion.db import DatabaseManager
from ingestion.pipeline import IngestionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_pipeline")


def run_single_ingestion(
    pipeline: IngestionPipeline,
    locations: Optional[List[str]] = None,
    forecast_days: int = 7,
) -> None:
    """Execute a single ingestion cycle across target locations."""
    logger.info("Executing scheduled ingestion cycle...")
    result = pipeline.run_live_forecast_ingestion(
        locations=locations,
        forecast_days=forecast_days,
    )
    print("\n" + "=" * 60)
    print(f"INGESTION RUN SUMMARY [ID: {result['run_id']}]")
    print(f"Timestamp: {result['timestamp']}")
    print(f"Overall Status: {result['status'].upper()}")
    print(
        f"Sources: {result['sources_succeeded']}/{result['sources_attempted']} succeeded"
    )
    print("-" * 60)
    for loc_id, summary in result["locations"].items():
        status_symbol = "OK" if summary["status"] == "full" else "WARN" if summary["status"] == "degraded" else "ERR"
        print(
            f"[{status_symbol}] {loc_id:15s}: {summary['succeeded']}/{summary['attempted']} sources OK"
        )
        if summary["errors"]:
            for err in summary["errors"]:
                print(f"      -> {err}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or schedule Hybrid Forecast Blending ingestion pipeline."
    )
    parser.add_argument(
        "--locations",
        nargs="+",
        choices=list(TARGET_LOCATIONS.keys()),
        default=None,
        help="Target locations to ingest (defaults to all 10 locations).",
    )
    parser.add_argument(
        "--forecast-days",
        type=int,
        default=7,
        help="Number of forecast days to pull (default: 7).",
    )
    parser.add_argument(
        "--schedule-hours",
        type=int,
        default=None,
        help="If set, runs repeatedly every N hours using schedule loop.",
    )
    parser.add_argument(
        "--archive-location",
        type=str,
        choices=list(TARGET_LOCATIONS.keys()),
        default=None,
        help="Ingest historical archive data for this location.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD) for historical archive ingestion.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD) for historical archive ingestion.",
    )

    args = parser.parse_args()

    db_manager = DatabaseManager()
    pipeline = IngestionPipeline(db_manager=db_manager)

    # Historical archive ingestion mode
    if args.archive_location:
        if not args.start_date or not args.end_date:
            logger.error("--start-date and --end-date are required for archive ingestion.")
            sys.exit(1)
        res = pipeline.ingest_archive_for_location(
            location_id=args.archive_location,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        print(f"Archive ingestion result for {args.archive_location}: {res}")
        return

    # Scheduled mode
    if args.schedule_hours:
        logger.info(
            "Scheduling ingestion every %d hour(s). Starting first cycle immediately.",
            args.schedule_hours,
        )
        run_single_ingestion(pipeline, args.locations, args.forecast_days)
        schedule.every(args.schedule_hours).hours.do(
            run_single_ingestion, pipeline, args.locations, args.forecast_days
        )
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Pipeline scheduler stopped by user.")
            return

    # Default: one-off run
    run_single_ingestion(pipeline, args.locations, args.forecast_days)


if __name__ == "__main__":
    main()
