"""District Operational Alert Bulletin Generator (Module 6 & Phase 6).

Generates official plain-language operational disaster advisories for local authorities,
NDRF/SDMA units, and citizens across multiple regional languages:
- English (Universal National Operational Standard)
- Hindi (Plains / Orographic / Arid: Delhi, Shimla, Jaisalmer)
- Marathi (Konkan / Coastal: Mumbai)
- Tamil (Coromandel Coastal: Chennai)
- Bengali (Deltaic / Eastern: Kolkata)
"""

from datetime import datetime
import logging
from typing import Any, Dict, List, Optional

from alerts.templates import (
    ACTION_ADVISORIES,
    ALERT_ACTION_TERMS,
    IMPACT_TEMPLATES,
)
from extremes.detector import HazardAlert
from ingestion.config import TARGET_LOCATIONS

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "हिन्दी (Hindi)",
    "mr": "मराठी (Marathi)",
    "ta": "தமிழ் (Tamil)",
    "bn": "বাংলা (Bengali)",
}

# Preferred regional language by location
LOCATION_LANGUAGE_MAP = {
    "delhi": ["en", "hi"],
    "mumbai": ["en", "mr", "hi"],
    "chennai": ["en", "ta"],
    "kolkata": ["en", "bn", "hi"],
    "bengaluru": ["en", "ta", "hi"],
    "shimla": ["en", "hi"],
    "jaisalmer": ["en", "hi"],
    "guwahati": ["en", "hi", "bn"],
    "nagpur": ["en", "mr", "hi"],
    "thiruvananthapuram": ["en", "ta"],
}


class DistrictAlertGenerator:
    """Generates official operational alert bulletins from extreme weather detector outputs."""

    @staticmethod
    def generate_bulletin(
        alert: HazardAlert,
        language: str = "en",
        issued_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a complete district advisory bulletin for a given hazard alert and language.
        
        Args:
            alert: HazardAlert dataclass produced by ExtremeDetector.
            language: Language code ('en', 'hi', 'mr', 'ta', 'bn').
            issued_at: Optional timestamp string. Defaults to current UTC time.
            
        Returns:
            Dictionary with formatted title, color code, synopsis, impact, actionable advice, and metadata.
        """
        lang = language if language in SUPPORTED_LANGUAGES else "en"
        issue_time = issued_at or datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        hazard_type = alert.hazard_type.upper()
        if hazard_type not in ["RAINFALL", "HEATWAVE", "WIND"]:
            hazard_type = "RAINFALL"

        level = alert.alert_level.upper()
        action_title = ALERT_ACTION_TERMS.get(level, {}).get(lang, f"ALERT: {level}")
        
        # Impact text
        impact_text = (
            IMPACT_TEMPLATES.get(hazard_type, {})
            .get(level, {})
            .get(lang, alert.description)
        )

        # Actionable advice
        action_text = (
            ACTION_ADVISORIES.get(hazard_type, {})
            .get(level, {})
            .get(lang, alert.action_advisory)
        )

        # Header titles by language
        header_map = {
            "en": f"DISTRICT DISASTER MANAGEMENT ADVISORY — {alert.station_name.upper()} ({alert.state.upper()})",
            "hi": f"जिला आपदा प्रबंधन मौसम चेतावनी — {alert.station_name} ({alert.state})",
            "mr": f"जिल्हा आपत्ती व्यवस्थापन हवामान इशारा — {alert.station_name} ({alert.state})",
            "ta": f"மாவட்ட பேரிடர் மேலாண்மை வானிலை எச்சரிக்கை — {alert.station_name} ({alert.state})",
            "bn": f"জেলা বিপর্যয় মোকাবিলা আবহাওয়া সতর্কতা — {alert.station_name} ({alert.state})",
        }

        severity_colors = {
            "RED": "#FF4B4B",
            "ORANGE": "#FFA500",
            "YELLOW": "#FFD700",
            "GREEN": "#28A745",
        }

        badge_color = severity_colors.get(level, "#6c757d")

        return {
            "location_id": alert.location_id,
            "station_name": alert.station_name,
            "state": alert.state,
            "topography": alert.topography,
            "hazard_type": hazard_type,
            "alert_level": level,
            "badge_color": badge_color,
            "language": lang,
            "language_name": SUPPORTED_LANGUAGES[lang],
            "header": header_map.get(lang, header_map["en"]),
            "action_term": action_title,
            "category_name": alert.category_name,
            "peak_metric": f"{alert.peak_value} {alert.unit}",
            "peak_time": alert.peak_time,
            "model_consensus": f"{alert.model_consensus_pct:.0f}%",
            "synopsis": alert.description,
            "impact_advisory": impact_text,
            "actionable_instructions": action_text,
            "issued_at": issue_time,
            "issuing_authority": "Hybrid AI–NWP Forecast Blending System | SIH PS 26081 (MoES / NCMRWF Guidelines)",
        }
