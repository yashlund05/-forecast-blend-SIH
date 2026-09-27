# Task Backlog

Work through phases in order. Each phase should leave the project in a runnable state — do not
move to the next phase with a broken build.

## Phase 0 — Setup
- [x] Initialize repo with folder structure from ARCHITECTURE.md
- [x] Set up Python env + requirements.txt (see TECH_STACK.md)
- [x] Confirm Open-Meteo endpoints respond for one test location (manual curl/script check)

## Phase 1 — Ingestion (Module 1)
- [x] Write API client for multi-model forecast endpoint (GFS, ICON, IFS)
- [x] Write API client for AIFS endpoint, handle its 6-hourly resolution explicitly
- [x] Write API client for ensemble endpoint
- [x] Write API client for historical/archive endpoint
- [x] Normalize all responses into one common schema
- [x] Add per-source error handling (one source failing must not crash ingestion)
- [x] Store raw + normalized data locally (SQLite or parquet)
- [x] Add scheduler entry point (`run_pipeline.py`) — even if run manually for now, structure it
      as if scheduled

**Milestone check**: can pull and store forecasts for all 10 locations from all sources.

## Phase 2 — Baseline blend + minimal dashboard (MVP)
- [ ] Compute simple weighted average blend (start with equal weights if skill engine isn't
      ready yet — get the plumbing working first)
- [ ] Build minimal Streamlit (or chosen frontend) view: pick a location, show individual model
      forecasts + blended forecast on one chart
- [ ] Confirm this runs end-to-end live (not from a static/cached file) before moving on

**Milestone check**: this is your fallback demo if nothing else finishes — must work standalone.

## Phase 3 — Skill/Weight Engine (Module 2)
- [ ] Implement per-location, per-season, per-lead-time error computation against historical
      archive ground truth
- [ ] Convert error to inverse-error weights
- [ ] Assign topography class per location (coastal/arid/hill/plains/deltaic)
- [ ] Replace Phase 2's equal-weight blend with these computed weights
- [ ] Build the weight-map visualization on the dashboard map

## Phase 4 — Verification Module (Module 5) — HIGH PRIORITY, do not skip or rush
- [ ] Define and hardcode the exact date ranges: training period / weight-calibration period /
      scoring period — no overlap, documented in code comments
- [ ] Compute RMSE/MAE for blend vs. each individual source over the scoring period
- [ ] Implement POD, FAR, CSI, ETS for extreme-event detection
- [ ] Add a verification panel to the dashboard showing all of the above, including at least one
      case where the blend does not win
- [ ] Write up the methodology in plain language for the pitch/demo script

## Phase 5 — Extreme Weather Module (Module 4)
- [ ] Source real IMD threshold definitions for heavy rainfall/heatwave/high wind (or mark
      placeholders clearly if not yet verified)
- [ ] Apply thresholds to blended forecast, flag/score events
- [ ] Identify one real historical extreme event in the archive window
- [ ] Build the case-study comparison: naive average vs. blend vs. actual observed value, with
      numbers
- [ ] Add extreme alert panel to dashboard

## Phase 6 — District alerts (differentiator)
- [ ] Pick 2+ locations/climate zones for alert generation
- [ ] Build templated alert strings (English + at least 1 regional language)
- [ ] Wire alert generation to the extreme module's output
- [ ] Display alerts on dashboard

## Phase 7 — Polish / stretch (only if time remains)
- [ ] Regime-gated ML model (LightGBM) replacing static inverse-error weights
- [ ] Explainability panel (simple "why this weight" trace, doesn't need full SHAP)
- [ ] CSV/JSON export button
- [ ] Scheduler status indicator on dashboard

## Pre-demo checklist
- [ ] Full pipeline runs live in front of judges without manual data injection
- [ ] Verification numbers are computed live/from real backtest, not hardcoded
- [ ] At least one honest limitation is ready to state if asked
- [ ] Open-Meteo / ECMWF / NOAA / DWD attribution is visible somewhere in the UI or README
