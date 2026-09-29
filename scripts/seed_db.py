#!/usr/bin/env python3
"""Seed script: backfills the last 90 days of data for all 10 stations and generates weights.

Run this once on a fresh clone before launching the dashboard:

    python scripts/seed_db.py

Expected runtime: 3–8 minutes depending on network latency (Open-Meteo free tier).

What it does:
1. Ingests live 7-day forecasts for all 10 stations (5 sources: GFS, ICON, ECMWF IFS,
   ECMWF AIFS, Google WeatherNext 2).
2. Backfills historical forecast + observation archive from Open-Meteo archive API
   for the last 90 days at all 10 stations.  This provides enough data for the
   WeightEngine to calibrate meaningful inverse-error weights.
3. Runs WeightEngine.generate_and_save_weights() to populate the model_weights table
   that powers Tab 2 (weight maps) and Tab 3 (verification) in the dashboard.

Note: The held-out verification test period (2024-07-01 to 2024-08-31) is pre-seeded
separately by the VerificationEngine on first call if its data is missing.  This script
targets the rolling 90-day calibration window only.
"""

import logging
import sys
from datetime import date, timedelta

# Ensure project root is on path when running as `python scripts/seed_db.py`
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from ingestion.config import TARGET_LOCATIONS
from ingestion.db import DatabaseManager
from ingestion.pipeline import IngestionPipeline
from weighting.weights import WeightEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("seed_db")

SEED_DAYS = 90  # Rolling calibration window (keeps runtime reasonable)


def main() -> None:
    """Run the full seed sequence."""
    db = DatabaseManager()
    pipeline = IngestionPipeline(db_manager=db)

    # ── Step 1: Live forecast ingestion ──────────────────────────────────────
    logger.info("Step 1/3: Ingesting live 7-day forecasts for all 10 stations …")
    result = pipeline.run_live_forecast_ingestion(forecast_days=7)
    succeeded = result["sources_succeeded"]
    attempted = result["sources_attempted"]
    logger.info(
        "Live ingestion complete: %d/%d sources succeeded.", succeeded, attempted
    )
    if result["status"] == "failed":
        logger.warning(
            "Live ingestion failed — dashboard Tab 1 will show 'no data' until "
            "forecasts are available.  Continuing with archive backfill …"
        )

    # ── Step 2: Historical archive backfill ──────────────────────────────────
    end_date = date.today()
    start_date = end_date - timedelta(days=SEED_DAYS)
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    logger.info(
        "Step 2/3: Backfilling %d-day historical archive (%s → %s) for all stations …",
        SEED_DAYS,
        start_str,
        end_str,
    )

    errors: list[str] = []
    for i, loc_id in enumerate(TARGET_LOCATIONS, 1):
        logger.info(
            "  [%d/%d] Archiving %s …", i, len(TARGET_LOCATIONS), loc_id
        )
        try:
            res = pipeline.ingest_archive_for_location(
                location_id=loc_id,
                start_date=start_str,
                end_date=end_str,
            )
            if res.get("status") != "ok":
                errors.append(f"{loc_id}: {res}")
        except Exception as exc:  # noqa: BLE001
            logger.error("  Archive ingestion failed for %s: %s", loc_id, exc)
            errors.append(f"{loc_id}: {exc}")

    if errors:
        logger.warning(
            "%d station(s) had archive errors (blend will use fewer calibration points "
            "for those stations):\n  %s",
            len(errors),
            "\n  ".join(errors),
        )
    else:
        logger.info("Historical archive backfill complete for all 10 stations.")

    # ── Step 3: Weight generation ─────────────────────────────────────────────
    logger.info("Step 3/3: Computing and saving model weights to SQLite …")
    try:
        engine = WeightEngine(db_manager=db)
        weights_df = engine.generate_and_save_weights()
        logger.info(
            "Weight generation complete: %d weight records saved.", len(weights_df)
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Weight generation failed: %s\n"
            "The dashboard will use equal weights as a fallback until this succeeds.",
            exc,
        )
        sys.exit(1)

    print("\n" + "=" * 60)
    print("SEED COMPLETE ✓")
    print(f"  Historical window : {start_str} → {end_str} ({SEED_DAYS} days)")
    print(f"  Stations seeded   : {len(TARGET_LOCATIONS)}")
    print(f"  Weight rows saved : {len(weights_df)}")
    print("  Next step: streamlit run dashboard/app.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
