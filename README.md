# Hybrid AI–NWP Multi-Model Forecast Blending System

SIH Problem Statement 26081 | Ministry of Earth Sciences / NCMRWF | Theme: Disaster Management

A hybrid AI–NWP multi-model forecast blending system that dynamically pulls live operational forecasts from Open-Meteo (NOAA GFS, DWD ICON, ECMWF IFS, and ECMWF AIFS), blends them using adaptive regional and seasonal inverse-error weights, flags extreme weather against verified IMD operational thresholds, and generates actionable multilingual district disaster bulletins.

**Status**: Hackathon prototype, feature-complete through Phase 7 (all tasks in `docs/TASKS.md` checked, 34/34 tests passing).

---

## What's Implemented

The system implements all six core architectural modules and two key differentiators:

- **Module 1 — Ingestion Pipeline**: Resilient multi-source fetcher (GFS, ICON, ECMWF IFS, ECMWF AIFS) with schema normalization, SQLite storage, synoptic 6-hourly scheduler, and graceful per-source failure degradation.
- **Module 2 — Skill & Weighting Engine**: Historical error backtesting computing quadratic inverse-error weights ($w_m \propto 1/\text{RMSE}_m^2$) conditioned on station, season, and variable, with low-sample fallbacks and audit-ready explainability traces.
- **Module 3 — Blending Engine**: Pluggable weighted-average blending with missing-model renormalization, min–max ensemble spread envelopes, and an optional regime-gated GBDT blending upgrade.
- **Module 4 — Extreme Weather Module**: Multi-hazard detection using verified India Meteorological Department (IMD) thresholds for 24h rainfall, heatwaves by topography, and distinct paths for convective squalls vs. synoptic gales, plus verifiable historical extreme-event backtesting.
- **Module 5 — Verification Engine**: Non-overlapping train/calibrate/test backtesting against reanalysis ground truth, computing RMSE/MAE reductions, contingency skill scores (POD, FAR, CSI, ETS), and transparent failure case auditing.
- **Module 6 — Dashboard & Multilingual Alerts**: 4-tab Streamlit operations dashboard featuring interactive forecast curves, national weight maps, "Why This Weight?" explainability panels, test-split verification charts, and official district warning bulletins in 5 languages (English, Hindi, Marathi, Tamil, Bengali).

### Key Differentiators
- **Real Live Data, Zero Fabrication**: Forecasts and reanalysis benchmarks are retrieved live from Open-Meteo APIs. No skill scores, case studies, or verification numbers are hardcoded or simulated.
- **Strict Leak-Free Time-Split & Honest Limitation Reporting**: Training (`2021-09-01` to `2024-04-30`) and test (`2024-07-01` to `2024-08-31`) windows are strictly non-overlapping. The verification dashboard reports all edge cases where individual physical models outperformed the blend rather than filtering them out.
- **Evidence-Backed, Qualified Claim**: Blending demonstrably helps for precipitation (**+7.81% vs. naive**, RMSE 1.076 mm vs. 1.167 mm) and wind speed (**+6.81% vs. naive**, **+9.16% vs. ECMWF IFS**, RMSE 2.728 km/h vs. 3.002 km/h, $p < 0.05$); for temperature, results favor using ECMWF IFS directly (IFS achieves **0.794 °C** vs. blend **0.833 °C**; 90% CI on difference excludes zero), as IFS's 4D-Var data assimilation leaves virtually no headroom for linear weight combinations with lower-resolution models.

---

## Empirical Verification Results (Held-Out Monsoon Test Split)

Evaluated across **43,920 point-predictions** (14,640 per variable across 10 national stations) during the reserved test period (`2024-07-01` to `2024-08-31`):

| Variable | NOAA GFS | DWD ICON | ECMWF IFS | Naive Blend | Learned Blend | Blend vs. Naive | Verdict vs. Best Single Model |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Temperature (2m)** | 2.129 °C | 1.520 °C | **0.794 °C** | 1.212 °C | 0.833 °C | **+31.22%** | **IFS alone statistically outperforms blend** (by 0.041 °C, $p < 0.05$) |
| **Precipitation** | 1.655 mm | 1.530 mm | 1.104 mm | 1.167 mm | **1.076 mm** | **+7.81%** | **+2.26% vs. IFS** (90% CI crosses zero — comparable) |
| **Wind Speed (10m)** | 4.735 km/h | 5.372 km/h | 3.002 km/h | 2.928 km/h | **2.728 km/h** | **+6.81%** | **Blend statistically beats best single model** (+9.16% vs. IFS, $p < 0.05$) |

- **Normalized Multi-Variate Skill Score**: **+15.28%** (unitless arithmetic mean of relative error reductions vs. naive averaging; macro sample-weighted reduction is **+14.13%**).
- **No Unit-Mixed RMSE**: Averaging RMSE across heterogeneous units (°C, mm, km/h) is mathematically invalid and has been removed from all reports in favor of normalized skill scoring and per-variable evaluation.
- For complete 1,000-resample bootstrap confidence intervals, full 16-case failure logs, and IMD SOP citations, see [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md).

---

## Setup & Run

### 1. Installation & Environment Setup
Clone the repository, create a virtual environment, and install required dependencies:
```bash
git clone <repo-url>
cd "forecast blend SIH"

python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

> **API Key Setup**: No API key is required. Open-Meteo endpoints used by this system are open access under CC BY 4.0. No `.env` configuration is needed to run out of the box.

### 2. Running Ingestion
The pipeline can be executed in multiple operational modes:
```bash
# A. One-off live ingestion for all 10 national stations (GFS, ICON, IFS, AIFS)
python run_pipeline.py

# B. One-off ingestion for specific target locations
python run_pipeline.py --locations mumbai delhi shimla

# C. Operational daemon mode (runs automated ingestion cycle every 6 hours)
python run_pipeline.py --schedule-hours 6

# D. Historical reanalysis observations backfill (3-year archive for training & evaluation)
python ingestion/backfill.py
```

### 3. Launching the Dashboard
Start the interactive Streamlit dashboard:
```bash
streamlit run dashboard/app.py
```
The dashboard will open at `http://localhost:8501`, providing access to:
- **Tab 1**: Live Dynamic Forecast Blend, model spread envelope, and CSV/JSON data export.
- **Tab 2**: National Model Weight Maps & "Why This Weight?" explainability panel.
- **Tab 3**: Held-Out Test Period Verification, contingency scores, failure case audit, and GBDT regime-gating evaluation.
- **Tab 4**: Extreme Weather Intelligence, Delhi/Mumbai/Jaisalmer case studies, and multilingual district alert bulletins.

### 4. Running the Test Suite
Run the comprehensive test suite covering all modules, IMD threshold citations, and anti-leakage guards:
```bash
pytest -v
```

---

## Docs
- `docs/PRD.md` — what we're building and why
- `docs/ARCHITECTURE.md` — system modules and data flow
- `docs/DATA_SOURCES.md` — API endpoints, variables, locations
- `docs/TECH_STACK.md` — libraries and tools
- `docs/TASKS.md` — phased build checklist
- `AGENTS.md` — rules for AI coding agents working on this repo

## Operational Data Source Status & NCMRWF Roadmap
**Current data source**: Open-Meteo (public reanalysis-backed archive + live multi-model API). Not yet using NCMRWF's own operational products (IMDAA/MERA reanalysis, IMD gridded station observations). Ingesting IMDAA/MERA directly is the natural next step for a fully NCMRWF-native pipeline; the architecture's pluggable ingestion clients (`ingestion/clients.py`) are designed to make this a contained change.

---

## Data attribution
Weather data via Open-Meteo (CC BY 4.0), sourced from ECMWF, NOAA, DWD and other national weather services. Attribution required in any public deployment.

