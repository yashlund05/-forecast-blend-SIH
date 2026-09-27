"""Operational Ingestion Scheduler (Module 1 & Phase 7).

Coordinates automated operational forecast ingestion aligned with standard
synoptic NWP cycles (00:00, 06:00, 12:00, 18:00 UTC).
Tracks scheduler health, next scheduled cycle, and daemon status.
"""

from datetime import datetime, timezone, timedelta
import logging
from typing import Any, Dict, Optional

from ingestion.db import DatabaseManager
from ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)

# Standard synoptic run hours (UTC) when NWP centers release model outputs
SYNOPTIC_CYCLES_UTC = [0, 6, 12, 18]

# Dissemination lag: Global models typically become accessible via Open-Meteo ~3.5 hours after run time
MODEL_DISSEMINATION_LAG_HOURS = 3.5


class IngestionScheduler:
    """Monitors and coordinates operational ingestion schedules and status."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None) -> None:
        self.db = db_manager or DatabaseManager()

    def get_next_scheduled_run(self, now: Optional[datetime] = None) -> datetime:
        """Calculate the next scheduled operational ingestion cycle in UTC."""
        ref = now or datetime.now(timezone.utc)
        current_hour = ref.hour + ref.minute / 60.0

        for cycle_h in SYNOPTIC_CYCLES_UTC:
            target_h = cycle_h + MODEL_DISSEMINATION_LAG_HOURS
            if current_hour < target_h:
                hour_part = int(target_h)
                min_part = int((target_h - hour_part) * 60)
                return ref.replace(hour=hour_part, minute=min_part, second=0, microsecond=0)

        # If past all cycles today, next cycle is tomorrow's first cycle
        first_cycle = SYNOPTIC_CYCLES_UTC[0] + MODEL_DISSEMINATION_LAG_HOURS
        hour_part = int(first_cycle)
        min_part = int((first_cycle - hour_part) * 60)
        tomorrow = ref + timedelta(days=1)
        return tomorrow.replace(hour=hour_part, minute=min_part, second=0, microsecond=0)

    def get_status(self) -> Dict[str, Any]:
        """Return operational scheduler status, last execution, and countdown."""
        now_utc = datetime.now(timezone.utc)
        history = self.db.get_pipeline_history(limit=1)

        last_run_iso = None
        last_status = "AWAITING RUN"
        last_sources_ok = 0
        last_sources_total = 0

        if not history.empty:
            last_record = history.iloc[0]
            last_run_iso = last_record["run_timestamp"]
            last_status = str(last_record["status"]).upper()
            last_sources_ok = int(last_record.get("sources_succeeded", 0))
            last_sources_total = int(last_record.get("sources_attempted", 0))

        next_run = self.get_next_scheduled_run(now_utc)
        time_to_next = next_run - now_utc
        minutes_to_next = max(0, int(time_to_next.total_seconds() // 60))
        hours_to_next = minutes_to_next // 60
        mins_rem = minutes_to_next % 60

        return {
            "status": "ACTIVE_CRON",
            "cycle_interval_hours": 6,
            "synoptic_cycles": "00:00, 06:00, 12:00, 18:00 UTC",
            "last_run_iso": last_run_iso,
            "last_status": last_status,
            "last_sources_ok": last_sources_ok,
            "last_sources_total": last_sources_total,
            "next_run_iso": next_run.isoformat(),
            "next_run_display": next_run.strftime("%Y-%m-%d %H:%M UTC"),
            "countdown_str": f"{hours_to_next}h {mins_rem}m",
            "is_degraded": (last_status == "DEGRADED"),
            "is_healthy": (last_status in ["SUCCESS", "OK"]),
        }
