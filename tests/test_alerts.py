"""Unit tests for District Operational Alert Generation (Module 6 & Phase 6).

Covers:
- (a) Multilingual bulletin formatting (English, Hindi, Marathi, Tamil, Bengali).
- (b) Integration with HazardAlert objects.
- (c) Inclusion of actionable guidance and color coding.
"""

import pytest

from alerts.generator import DistrictAlertGenerator, SUPPORTED_LANGUAGES
from extremes.detector import HazardAlert


@pytest.fixture
def sample_orange_alert():
    return HazardAlert(
        location_id="mumbai",
        station_name="Mumbai",
        state="Maharashtra",
        topography="coastal",
        hazard_type="RAINFALL",
        alert_level="ORANGE",
        category_name="Very Heavy Rain",
        peak_value=145.2,
        unit="mm/24h",
        peak_time="2024-07-12T14:00",
        model_consensus_pct=100.0,
        description="Expected 24h accumulation of 145.2 mm with intense convective surges.",
        action_advisory="Avoid waterlogged underpasses and check traffic updates.",
    )


def test_multilingual_bulletin_structure(sample_orange_alert):
    """Verify generated bulletins contain all required operational keys across languages."""
    for lang in ["en", "hi", "mr", "ta", "bn"]:
        bulletin = DistrictAlertGenerator.generate_bulletin(sample_orange_alert, language=lang)

        assert bulletin["location_id"] == "mumbai"
        assert bulletin["alert_level"] == "ORANGE"
        assert bulletin["badge_color"] == "#FFA500"
        assert bulletin["language"] == lang
        assert bulletin["language_name"] == SUPPORTED_LANGUAGES[lang]
        assert len(bulletin["header"]) > 5
        assert len(bulletin["action_term"]) > 5
        assert len(bulletin["impact_advisory"]) > 10
        assert len(bulletin["actionable_instructions"]) > 10
        assert "Hybrid AI–NWP" in bulletin["issuing_authority"]


def test_alert_severity_terms_distinct():
    """Verify distinct operational action terms for RED, ORANGE, YELLOW, GREEN."""
    levels = ["RED", "ORANGE", "YELLOW", "GREEN"]
    terms_en = [DistrictAlertGenerator.generate_bulletin(
        HazardAlert("delhi", "Delhi", "Delhi", "plains", "RAINFALL", lvl, "Cat", 10.0, "mm", "2024-07-31", 100.0, "Desc", "Adv"),
        language="en"
    )["action_term"] for lvl in levels]

    assert len(set(terms_en)) == 4
    assert "WARNING" in terms_en[0]
    assert "ALERT" in terms_en[1]
    assert "WATCH" in terms_en[2]
    assert "NO WARNING" in terms_en[3]
