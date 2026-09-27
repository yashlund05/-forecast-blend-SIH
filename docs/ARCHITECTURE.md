# Architecture — Hybrid AI–NWP Forecast Blending System

## System overview (6 modules)

```
[1. Ingestion] --> [2. Skill/Weight Engine] --> [3. Blending Engine] --> [4. Extreme Module]
                                                        |                       |
                                                        v                       v
                                              [5. Verification Module] --> [6. Dashboard/Alerts]
```

## Module 1 — Ingestion
- Pulls live forecasts from Open-Meteo for each target location: ECMWF IFS, ECMWF AIFS, NOAA GFS,
  DWD ICON, ECMWF ENS (ensemble spread).
- Pulls historical/archive data for the same locations for backtesting and weight training.
- Normalizes all sources into one common schema (see DATA_SOURCES.md for exact fields).
- Runs on a schedule (not manually triggered) — this satisfies the "operational workflow"
  requirement.
- Must fail gracefully per-source: if one model's API call fails, log it and continue with the
  remaining sources rather than crashing the whole pipeline.

## Module 2 — Skill/Weight Engine
- For each (location, topography class, season, lead-time bucket), computes each source's
  historical error (MAE/RMSE) against ground truth from the historical archive.
- Converts error into a weight (inverse-error weighting as the baseline method).
- Topography classes: coastal, arid, hill/orographic, plains, deltaic/cyclone-exposed — assign
  per location, used as a feature/bucket, not hardcoded into the blend logic.
- Output: a weight table (location x season x lead-time x source -> weight), which is directly
  the "model weight map" deliverable.
- Stretch: replace static inverse-error weighting with a trained gating model (e.g. LightGBM)
  that predicts weights from region/season/lead-time/regime features.

## Module 3 — Blending Engine
- Baseline: weighted average of source forecasts using Module 2's weights.
- Must NOT use simple unweighted averaging as the final method — the whole point is adaptive
  weighting; unweighted average is only a comparison baseline for Module 5.
- Stretch: quantile-preserving blending for extremes (do not mean-blend the tail of the
  distribution the same way as the center).

## Module 4 — Extreme Weather Module
- Applies thresholds for heavy rainfall / heatwave / high wind to the blended output.
- Thresholds must be sourced (cite the IMD definition used in a code comment) or explicitly
  labeled `# PLACEHOLDER - verify against IMD operational criteria` if not yet confirmed.
- Includes a backtest case study: pick one real historical extreme event in the archive window,
  show naive-average forecast vs. blended forecast vs. actual observed value.

## Module 5 — Verification Module
- Strict time split: training period, weight-calibration period, and scoring period must not
  overlap. Document the exact date ranges used in code and in the dashboard.
- Metrics: RMSE/MAE for rainfall/temp/wind; POD, FAR, CSI, ETS for extreme-event detection.
- Must display metrics for the blended system AND each individual source, for honest comparison.
- Must surface at least one case where the blend does not outperform (do not hide failure cases).

## Module 6 — Dashboard / Alerting Layer
- Map view of India (or target region) with weight-map overlay (color by dominant trusted
  source per location/season).
- Time-series panel: individual model forecasts vs. blended forecast vs. actual (for backtest
  periods) and vs. live forecast (for current period).
- Extreme weather alert panel with confidence/probability.
- District-level plain-language alert generator, at least one regional language, template-based
  (do not require a full translation model — templated strings with variable slots is enough).
- Export: CSV/JSON of blended forecast + skill metrics.
- Scheduler status indicator showing when the pipeline last ran (proves it's "operational,"
  not a one-shot script).

## Suggested folder structure
```
/data           - raw and processed forecast/archive data (gitignored if large)
/ingestion       - API clients per source, scheduler
/weighting       - skill computation, weight tables
/blending        - blending logic (baseline + stretch gating model)
/extremes        - threshold logic, case-study backtest script
/verification    - metrics computation, time-split enforcement
/dashboard       - frontend app
/alerts          - templated alert generation, language strings
/tests           - unit tests per module
/docs            - this file, PRD, data sources, tasks
```
