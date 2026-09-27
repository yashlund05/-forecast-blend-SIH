"""Unit tests for Extreme Weather Module (Module 4 & Phase 5).

Covers:
- (a) Verified IMD threshold classification for rainfall (24h and hourly), heatwaves (by topography), and winds.
- (b) ExtremeDetector alerts generation and compound hazard resolution.
- (c) CaseStudyEngine live execution on held-out test period events (Delhi, Mumbai, Jaisalmer).
- (d) Hard Rule 1 & Hard Rule 4: Real computed backtest numbers with zero data leakage.
"""

import pytest

from extremes.case_study import CaseStudyEngine
from extremes.detector import ExtremeDetector, HazardAlert
from extremes.thresholds import (
    HEAVY_RAIN_24H_THRESHOLD_MM,
    HEAVY_RAIN_HOURLY_THRESHOLD_MM,
    classify_heatwave,
    classify_hourly_rainfall,
    classify_rainfall_24h,
    classify_wind,
    resolve_compound_alert,
)
from verification.guards import assert_strictly_test_period


def test_imd_rainfall_threshold_classification():
    """Verify official IMD 24h rainfall classifications and alert levels."""
    # 1. Very light (< 2.5 mm)
    cat, alert = classify_rainfall_24h(1.2)
    assert "Very Light" in cat
    assert alert == "GREEN"

    # 2. Moderate (15.6 - 64.4 mm)
    cat, alert = classify_rainfall_24h(35.0)
    assert cat == "Moderate Rain"
    assert alert == "GREEN"

    # 3. Heavy Rain (64.5 - 115.5 mm)
    cat, alert = classify_rainfall_24h(64.5)
    assert cat == "Heavy Rain"
    assert alert == "YELLOW"

    cat, alert = classify_rainfall_24h(100.0)
    assert cat == "Heavy Rain"
    assert alert == "YELLOW"

    # 4. Very Heavy Rain (115.6 - 204.4 mm)
    cat, alert = classify_rainfall_24h(115.6)
    assert cat == "Very Heavy Rain"
    assert alert == "ORANGE"

    cat, alert = classify_rainfall_24h(150.0)
    assert cat == "Very Heavy Rain"
    assert alert == "ORANGE"

    # 5. Extremely Heavy Rain (>= 204.5 mm)
    cat, alert = classify_rainfall_24h(220.0)
    assert cat == "Extremely Heavy Rain"
    assert alert == "RED"


def test_imd_heatwave_thresholds_by_topography():
    """Verify IMD heatwave thresholds differentiated across plains, coastal, and hills."""
    # Plains: Base 40°C, Orange at 43°C, Red at 45°C
    cat, alert = classify_heatwave(39.0, topography="plains")
    assert alert == "GREEN"

    cat, alert = classify_heatwave(41.5, topography="plains")
    assert alert == "YELLOW"

    cat, alert = classify_heatwave(43.5, topography="plains")
    assert alert == "ORANGE"

    cat, alert = classify_heatwave(46.0, topography="plains")
    assert alert == "RED"

    # Coastal (e.g. Mumbai, Chennai): Base 37°C
    cat, alert = classify_heatwave(36.5, topography="coastal")
    assert alert == "GREEN"

    cat, alert = classify_heatwave(38.0, topography="coastal")
    assert alert == "YELLOW"

    cat, alert = classify_heatwave(42.5, topography="coastal")
    assert alert == "RED"

    # Hills (e.g. Shimla): Base 30°C
    cat, alert = classify_heatwave(29.0, topography="hill")
    assert alert == "GREEN"

    cat, alert = classify_heatwave(31.0, topography="hill")
    assert alert == "YELLOW"

    cat, alert = classify_heatwave(35.5, topography="hill")
    assert alert == "RED"


def test_imd_wind_threshold_classification():
    """Verify IMD squall and gale criteria (IMD SOP Ch 6 Sec 6.3.1)."""
    cat, alert = classify_wind(25.0)
    assert alert == "GREEN"

    cat, alert = classify_wind(40.5)
    assert "Strong" in cat
    assert alert == "YELLOW"

    cat, alert = classify_wind(55.0)
    assert "Squall" in cat
    assert alert == "ORANGE"

    cat, alert = classify_wind(70.0)
    assert "Gale" in cat or "Squall" in cat
    assert alert == "RED"

    cat, alert = classify_wind(90.0)
    assert "Severe Gale" in cat
    assert alert == "RED"


def test_resolve_compound_alert_priority():
    """Ensure priority order: RED > ORANGE > YELLOW > GREEN."""
    assert resolve_compound_alert(["GREEN", "YELLOW"]) == "YELLOW"
    assert resolve_compound_alert(["YELLOW", "ORANGE", "GREEN"]) == "ORANGE"
    assert resolve_compound_alert(["ORANGE", "RED", "YELLOW"]) == "RED"
    assert resolve_compound_alert(["GREEN"]) == "GREEN"


def test_case_study_delhi_2024_07_31():
    """Verify live calculation of the July 31, 2024 Delhi cloudburst case study.
    
    Hard Rule 1: Every number is derived from database records.
    Hard Rule 4: Confined to held-out test split.
    """
    engine = CaseStudyEngine()
    result = engine.run_case_study("delhi_2024_07_31")

    assert result.event_id == "delhi_2024_07_31"
    assert result.location_id == "delhi"
    assert result.date == "2024-07-31"

    # Assert test split boundaries
    assert_strictly_test_period(result.date, result.date)

    # Observed precipitation was 144.6 mm
    assert result.observed_total == pytest.approx(144.6, abs=0.5)

    metrics = result.metrics_by_source
    assert "learned_blend" in metrics
    assert "naive_blend" in metrics
    assert "ecmwf_ifs" in metrics
    assert "gfs" in metrics
    assert "icon" in metrics

    # Learned blend must have lower RMSE than naive blend
    assert metrics["learned_blend"]["rmse"] < metrics["naive_blend"]["rmse"]
    assert metrics["learned_blend"]["rmse"] == pytest.approx(3.797, abs=0.01)

    # Alert classification match
    assert result.alert_classification["observed"]["alert_level"] == "ORANGE"
    assert result.alert_classification["learned_blend"]["alert_level"] == "ORANGE"
    assert result.alert_classification["naive_blend"]["alert_level"] == "YELLOW"

    # Check hourly series shape
    assert len(result.hourly_df) == 24


def test_case_study_jaisalmer_heatwave():
    """Verify Thar Desert heat spell case study on 2024-07-16."""
    engine = CaseStudyEngine()
    result = engine.run_case_study("jaisalmer_2024_07_16")

    assert result.observed_peak == pytest.approx(42.0, abs=0.5)
    assert result.metrics_by_source["learned_blend"]["peak"] == pytest.approx(42.05, abs=0.5)
    assert result.metrics_by_source["learned_blend"]["rmse"] < result.metrics_by_source["naive_blend"]["rmse"]


def test_threshold_citation_or_unverified_flag():
    """Citation & Labeling Audit Test:
    
    Every single threshold constant in extremes/thresholds.py MUST have either:
    1. An in-session verified citation referencing the IMD SOP (March 2021) with Chapter/Section/Page, OR
    2. An explicit UNVERIFIED flag ('# UNVERIFIED - could not confirm against source, review before demo').
    
    Zero unverified thresholds may be presented as verified.
    """
    import re
    from pathlib import Path

    thresholds_file = Path(__file__).resolve().parent.parent / "extremes" / "thresholds.py"
    assert thresholds_file.exists(), "extremes/thresholds.py must exist"

    lines = thresholds_file.read_text(encoding="utf-8").splitlines()

    constant_pattern = re.compile(r"^(IMD_[A-Z0-9_]+|HEAVY_[A-Z0-9_]+|HEATWAVE_[A-Z0-9_]+|HIGH_[A-Z0-9_]+)\s*[:=]")

    checked_constants = []
    for idx, line in enumerate(lines, start=1):
        match = constant_pattern.match(line.strip())
        if match:
            const_name = match.group(1)
            # Check comment on the same line or immediate previous line
            same_line_comment = line[line.find("#"):] if "#" in line else ""
            prev_line_comment = lines[idx - 2] if idx >= 2 and "#" in lines[idx - 2] else ""
            combined_context = f"{same_line_comment} {prev_line_comment}"

            has_verified = "VERIFIED:" in combined_context and "IMD SOP" in combined_context
            has_unverified = "UNVERIFIED" in combined_context

            assert has_verified or has_unverified, (
                f"Threshold constant '{const_name}' at line {idx} in extremes/thresholds.py violates rule: "
                f"must have either a verified citation ('VERIFIED: IMD SOP (March 2021) ...') "
                f"or an explicit '# UNVERIFIED' flag."
            )
            checked_constants.append(const_name)

    assert len(checked_constants) >= 12, f"Expected at least 12 threshold constants checked, got {len(checked_constants)}"

