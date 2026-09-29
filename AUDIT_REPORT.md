# Pre-Submission Audit Report: Hybrid AI–NWP Forecast Blending System
**SIH Problem Statement 26081** | Ministry of Earth Sciences / NCMRWF | Theme: Disaster Management  
**Date of Audit**: September 29, 2026 | **Auditor**: Autonomous Pre-Submission Auditor (Read-Only Pass)

---

## Executive Summary

A comprehensive, read-only technical audit of the repository was performed across all 10 evaluation criteria required for final submission. The audit inspected source code, execution pipelines, SQLite schemas, test suites, dashboard rendering logic, and documentation.

| # | Evaluation Item | Status | Key Evidence / Primary Observation |
| :-: | :--- | :---: | :--- |
| **1** | **Lead-Time Weighting** | **PASS** | `weighting/skill.py`, `weighting/weights.py`, SQLite `model_weights` table, Tab 2 selectbox |
| **2** | **Regime Conditioning** | **PARTIAL** | Implemented from ingested precip (`blending/regime.py`), but live Tab 1 blending omits `obs_df`, falling back to `all_regimes` |
| **3** | **Second AI/ML Model** | **PARTIAL** | Google WeatherNext 2 is a real Open-Meteo API with 6h interpolation; included in live blend/charts, but absent from historical verification |
| **4** | **Spatial Weight Maps** | **PASS** | IDW grid exists, sovereign masked via `data/india_boundary.geojson`, 500 km cutoff enforced, UI warning banner visible |
| **5** | **NCMRWF/IMDAA Status** | **FAIL** | README lacks detailed explanation of IMDAA access constraints; `--ground-truth-source` CLI option is completely missing |
| **6** | **Number Consistency** | **FAIL** | Bug in `verification/metrics.py` line 215 overwrites weights with equal 0.20, causing code to compute 0.0% gain while docs claim +15.28% |
| **7** | **Claims vs. Implementation** | **PARTIAL** | Tab 3 metric cards are hardcoded strings; live forecast blend defaults to `all_regimes`; verification is 3-model not 5-model |
| **8** | **Leakage Safeguards** | **PASS** | Train/Calibrate/Test windows are strictly non-overlapping; runtime assertions guard splits; features are strictly backward-looking |
| **9** | **Reproducibility** | **PARTIAL** | `requirements.txt` has all packages, but `data/forecast_blend.sqlite3` is gitignored, and README lacks DB initialization steps |
| **10** | **Test Suite** | **PASS** | `pytest -v` collects 43 tests: **43 passed**, 0 failed, 0 skipped, 0 xfailed (duration: 167.49s) |

---

## Detailed Item-by-Item Findings

### 1. Lead-Time Weighting: PASS
- **Computed Per Bucket**: `weighting/skill.py` lines 19–20 defines `LEAD_TIME_BUCKETS = {"0-24h", "24-72h", "72-120h", "120h+", "all"}`. `SkillEngine.compute_skill_metrics()` and `WeightEngine.generate_and_save_weights()` partition observations by forecast horizon and compute inverse-error weights separately for each horizon.
- **Distinct Values in Storage**: Querying `SELECT DISTINCT lead_time_bucket FROM model_weights` in `data/forecast_blend.sqlite3` returns all 5 distinct values (`['0-24h', '120h+', '24-72h', '72-120h', 'all']`).
- **Dynamic Dashboard Behavior**: In `dashboard/app.py` (Tab 2), `map_lead_bucket = st.selectbox(...)` directly controls `weight_engine.get_dominant_model_map()`. Changing the bucket alters dominant models across stations (e.g. for Mumbai precipitation, Day 1 `0-24h` assigns dominant model `icon` with weight 31.5%, whereas Days 6–7 `120h+` shifts dominance to `ecmwf_ifs` with weight 21.5%).
- **Automated Verification**: Covered by unit test `tests/test_weighting.py::test_weights_differ_across_lead_time_buckets` (PASSED).

### 2. Regime Conditioning: PARTIAL
- **Variable Grounding**: Implemented in `blending/regime.py` (`classify_regime` and `classify_regime_series`). It uses rolling 7-day mean precipitation ($\ge 0.3\text{ mm/hr}$) and calendar month to classify states into `monsoon_active`, `monsoon_break`, and `off_season`. These derive exclusively from `CORE_VARIABLES` actually ingested.
- **Defensible Scope**: Cyclone regime detection was evaluated against TRAIN reanalysis (24,792 hourly steps per station) and dropped due to negligible sample prevalence (0.0% to 0.3%) and absence of surface pressure in `CORE_VARIABLES`. This decision is documented with citations in `blending/regime.py` lines 15–37.
- **Engine Default vs. Call-Site Limitation**:
  - In `WeightEngine.generate_and_save_weights()`, weights are conditioned on `regime` by default (populating `regime` column in `model_weights`).
  - In `ForecastBlendEngine.blend(forecasts_df, obs_df=None)` (`blending/engine.py` line 138/165), `obs_df` is an optional parameter defaulting to `None`.
  - In `dashboard/app.py` line 303 (Tab 1 live blending), `blend_engine.blend(forecasts_df)` is invoked **without passing `obs_df`**. Consequently, live operational blending falls back to `regime="all_regimes"` rather than applying real-time regime classification.
- **Dashboard Visibility**: Tab 2 includes a "Weather Regime" selectbox (`map_regime`) that updates weight maps. Tab 3 includes an expandable "Regime-Stratified Blend Performance" section backed by `verif_engine.evaluate_regime_stratified()`.

### 3. Second AI/ML Model: PARTIAL
- **Real Verified API**: `ingestion/clients.py` defines `WeatherNextClient` connecting to Open-Meteo's Ensemble API (`https://ensemble-api.open-meteo.com/v1/ensemble`) requesting `models="google_weathernext2_ensemble"`. A live API call in this session confirmed real responses with HTTP 200 and 64 ensemble members for temperature, precipitation, and wind speed.
- **Native Resolution Handling**: Handled in `ingestion/normalizer.py::normalize_weathernext()`. Reconciles native 6-hourly steps to 1-hourly target grids via linear interpolation and flags rows with `is_interpolated=1`.
- **Presence in Live Pipelines**: Included in `BLEND_MODELS` (`blending/engine.py`), `model_styles` (`dashboard/app.py`), and persisted to SQLite `model_weights` table.
- **Absence in Reanalysis Verification Split**: In `verification/metrics.py` line 176, the evaluation models list is explicitly restricted to `models = ["gfs", "icon", "ecmwf_ifs"]`. Open-Meteo does not provide historical hindcasts for Google WeatherNext 2 or ECMWF AIFS in its historical forecast archive API. Per `VERIFICATION_REPORT.md` lines 52–57, AI models receive baseline equal weighting shares (0.20) and are marked `low_confidence=True`, but are **not** evaluated in the blind backtest.

### 4. Spatial Weight Maps: PASS
- **IDW Formulation**: `weighting/weights.py::interpolate_weight_grid` provides a vectorized 2D NumPy inverse-distance-weighting interpolator ($p=2.0$, 28×28 regular grid spanning 8°N–36°N and 68°E–96°E).
- **Sovereign Border Masking**: `weighting/weights.py::mask_grid_to_india` tests cells against official Natural Earth sovereign boundaries committed to `data/india_boundary.geojson`. Cells over the Arabian Sea, Bay of Bengal, Pakistan, Nepal, and Tibet are set to `np.nan`.
- **Proximity Cutoff**: Named constant `MAX_INTERPOLATION_DISTANCE_KM = 500.0` ensures cells exceeding 500 km from any active station are masked out to avoid ungrounded geographic extrapolation.
- **Caveat Banner**: Rendered in `dashboard/app.py` Tab 2:
  > *"⚠️ Methodological Caveat: Continuous field is a coarse indicative visual interpolation derived from 10 national benchmark stations, strictly masked to India's boundary and limited to areas within 500 km of an operational station..."*
- **Unit Tests**: `test_interpolate_weight_grid_smoke`, `test_mask_grid_to_india_known_locations`, and `test_mask_grid_to_india_distance_cutoff` all pass in `tests/test_weighting.py`.

### 5. NCMRWF/IMDAA Status: FAIL
- **Vague Documentation**: `README.md` (lines 142–144) states: *"Current data source: Open-Meteo... Not yet using NCMRWF's own operational products (IMDAA/MERA reanalysis, IMD gridded station observations)."* It does not explain *what was attempted, why IMDAA was inaccessible* (institutional credentials, TDS server access, lack of public REST API, large GRIB2 archive volumes).
- **Missing CLI Option**: The prompt specifies: *"is there a working `--ground-truth-source` option?"* `git grep "ground-truth-source"` produces 0 occurrences. `run_pipeline.py` has no such argument.

### 6. Number Consistency: FAIL (Critical Bug Identified)
- **Severe Code vs. Report Discrepancy**:
  When `VerificationEngine.evaluate_test_period()` is executed directly from code:
  - `mean_imp_vs_naive_pct`: **0.0%**
  - `normalized_skill_score_pct`: **0.0%**
  - `rmse_blend`: Matches `rmse_naive` exactly across all 3 variables (Temperature: 1.212 °C, Precip: 1.167 mm, Wind: 2.928 km/h).
- **Discrepancy with Reports**:
  - `README.md` and `VERIFICATION_REPORT.md` report:
    - Temperature: Learned Blend **0.833 °C** vs. Naive 1.212 °C (**+31.22%**)
    - Precipitation: Learned Blend **1.076 mm** vs. Naive 1.167 mm (**+7.81%**)
    - Wind Speed: Learned Blend **2.728 km/h** vs. Naive 2.928 km/h (**+6.81%**)
    - Normalized Multi-Variate Skill Score: **+15.28%**
- **Discrepancy within Dashboard (Tab 3)**:
  - Metric headline cards (`dashboard/app.py` lines 881–912) display hardcoded strings (`"0.833 °C"`, `"1.076 mm"`, `"2.728 km/h"`).
  - The interactive grouped bar chart directly beneath them (`comp_fig`) plots `metrics_df["rmse_blend"]` against `metrics_df["rmse_naive"]`. In that chart, the "Dynamic Learned Blend" bar is **identical in height** to "Naive Equal Blend", showing 0% improvement!
- **Root Cause Analysis**:
  In `verification/metrics.py` lines 215–220:
  ```python
  learned_weights_df = self.db.get_model_weights(location_id=loc_id, season="monsoon")
  weights_by_var: Dict[str, Dict[str, float]] = {}
  if not learned_weights_df.empty:
      for var, grp in learned_weights_df.groupby("variable"):
          weights_by_var[str(var)] = dict(zip(grp["model"], grp["weight"]))
  ```
  `get_model_weights(location_id=loc_id, season="monsoon")` returns 100 rows per variable across all 5 lead-time buckets and 4 regimes. Grouping only by `variable` iterates through all buckets; the final bucket processed is an uncalibrated slice with equal weights `0.2000` for all 5 models.
  When normalized across the 3 evaluated models (`gfs`, `icon`, `ecmwf_ifs`), each model receives $0.20 / 0.60 = 0.3333$, collapsing the learned blend into an equal-weight naive blend.
  *Fix*: Pass `regime="all_regimes", lead_time_bucket="all"` to `get_model_weights()`.

### 7. Claims vs. Implementation: PARTIAL
- **Discrepancies Identified**:
  1. **Claim**: *"No skill scores, case studies, or verification numbers are hardcoded or simulated"* (README.md line 27).  
     **Reality**: In `dashboard/app.py` lines 881–912, Tab 3's headline cards are hardcoded strings (`"0.833 °C"`, `"1.076 mm"`, `"2.728 km/h"`), bypassing the computed values in `summary["var_summaries"]`.
  2. **Claim**: *"Regime-Conditioned Weighting (core feature, not stretch goal)... classified in real-time from rolling 7-day precipitation anomaly"* (README.md line 30).  
     **Reality**: In `dashboard/app.py` line 303, `blend_engine.blend(forecasts_df)` is called without `obs_df`, so real-time operational forecasting always uses `all_regimes` weights.
  3. **Claim**: *"5 blended model sources... all 5 participating models"* (README.md lines 18–25).  
     **Reality**: Only 3 models (`gfs`, `icon`, `ecmwf_ifs`) participate in the empirical verification backtest due to upstream API hindcast constraints.
  4. **Claim**: Pluggable reanalysis ground truth architecture.  
     **Reality**: `--ground-truth-source` does not exist as a CLI option.

### 8. Leakage: PASS
- **Temporal Split Boundaries**:
  - TRAIN: `2021-09-01` to `2024-04-30` (32 months)
  - CALIBRATE: `2024-05-01` to `2024-06-30` (2 months)
  - TEST: `2024-07-01` to `2024-08-31` (2 months)
  The partitions are strictly non-overlapping.
- **Assertion Enforcement**:
  - `weighting/skill.py::assert_valid_skill_date_range` raises `AssertionError` if training or calibration queries touch `2024-07-01` or later.
  - `verification/guards.py::assert_strictly_test_period` raises `AssertionError` if verification touches dates before `2024-07-01` or after `2024-08-31`.
  - Verified by tests `test_anti_leakage_assertion_strictly_guards_test_period` and `test_verification_anti_leakage_guard`.
- **Feature Construction**:
  - Regime detection uses a backward-looking 168-hour rolling precipitation window (`.rolling(168, min_periods=1)`), with no forward-looking data leakage.
  - Lead-time buckets are computed causally from `target_time - fetch_time`.

### 9. Reproducibility: PARTIAL
- **Dependencies**: All imported third-party libraries (`numpy`, `pandas`, `plotly`, `pytest`, `requests`, `schedule`, `sklearn`, `streamlit`, `httpx`, `pyarrow`, `python-dotenv`) are declared in `requirements.txt`.
- **Gitignore Gap**: `*.sqlite3` and `*.db` are listed in `.gitignore`. Consequently, `data/forecast_blend.sqlite3` is an untracked/ignored file. On a fresh clone, the repository contains no SQLite database.
- **Bootstrap Gap**: Running `python run_pipeline.py` on a clean clone only fetches live forecasts; it does not backfill reanalysis or generate `model_weights`. `python ingestion/backfill.py` fetches historical data, but there is no CLI script to run `WeightEngine.generate_and_save_weights()`. Tests like `test_stored_database_weights_integrity` will skip if cloned on a clean machine without an existing SQLite database.

### 10. Test Suite: PASS
- Executed: `python -m pytest tests/ -v`
- **Results**:
  - **43 passed** in 167.49s (0:02:47)
  - **0 failed**
  - **0 skipped**
  - **0 xfailed**
- Tests comprehensively cover: multilingual bulletins, weight normalization, missing-model fallbacks, IMD extreme-weather thresholds, live API ingestion, offline/dashboard parity, ML gating deterministic parity, anti-leakage guards, contingency scoring math, honest failure case detection, regime classification, and spatial IDW masking.

---

## Ranked Top 5 Issues Most Likely to Hurt Us in Front of Judges

### 1. Verification Engine Weight Lookup Collapses to Naive Blend (Critical Number Inconsistency)
- **Problem**: In `verification/metrics.py` line 215, `get_model_weights(location_id=loc_id, season="monsoon")` does not filter by regime or lead time, causing uncalibrated equal weights (`0.2000`) to overwrite learned weights. Code execution produces 0.0% improvement over naive, directly contradicting the +15.28% skill score claimed in `README.md` and `VERIFICATION_REPORT.md`.
- **Suggested Fix**: In `verification/metrics.py` line 215, add `regime="all_regimes", lead_time_bucket="all"` to the `get_model_weights()` query.

### 2. Dashboard Tab 3 Hardcoded Headline Cards Disagree with Live Verification Chart
- **Problem**: Headline cards in Tab 3 show static strings (`"0.833 °C"`, `"1.076 mm"`, `"2.728 km/h"`), but the interactive grouped bar chart directly below renders live `metrics_df`, showing the learned blend bar identical in height to the naive blend.
- **Suggested Fix**: Bind `vk1`–`vk4` in `dashboard/app.py` dynamically to `summary["var_summaries"][v]["rmse_blend"]` and `summary["normalized_skill_score_pct"]` once Issue #1 is resolved.

### 3. Missing SQLite Database on Clean Clone (`.gitignore` Exclusion)
- **Problem**: `data/forecast_blend.sqlite3` is gitignored (`*.sqlite3`), meaning judges cloning the repo will start with an empty database and cannot run offline verification or explore Tab 2/Tab 3 without re-running a lengthy multi-year ingestion backfill.
- **Suggested Fix**: Force-commit a compact, pre-seeded evaluation SQLite database (`git add -f data/forecast_blend.sqlite3`) and add a note in `README.md`.

### 4. Missing `--ground-truth-source` CLI Option and Incomplete NCMRWF Inaccessibility Explanation
- **Problem**: `README.md` mentions future NCMRWF IMDAA integration but fails to explain why IMDAA is currently inaccessible (lack of open REST API, registration requirements), and the requested `--ground-truth-source` CLI flag does not exist in `run_pipeline.py`.
- **Suggested Fix**: Add `--ground-truth-source {open-meteo-era5, imdaa-mock}` to `run_pipeline.py` and document specific NCMRWF institutional access barriers in `README.md`.

### 5. Live Tab 1 Forecast Blending Bypasses Real-Time Regime Detection
- **Problem**: In `dashboard/app.py` line 303, `blend_engine.blend(forecasts_df)` is called without `obs_df`, forcing real-time blending to fall back to `all_regimes` weights rather than the active synoptic regime.
- **Suggested Fix**: Pass `obs_df=db_manager.get_latest_observations(selected_loc_id)` into `blend_engine.blend()` in `dashboard/app.py`.
