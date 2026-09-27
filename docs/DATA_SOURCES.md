# Data Sources

## Primary provider: Open-Meteo
Free, no API key required for non-commercial use. Data licensed under CC BY 4.0 — you MUST
credit Open-Meteo and the underlying national weather services (ECMWF, NOAA, DWD, etc.) in the
dashboard footer and/or README.

Base docs: https://open-meteo.com

### Operational Data Source Status & NCMRWF Roadmap
**Current data source**: Open-Meteo (public reanalysis-backed archive + live multi-model API). Not yet using NCMRWF's own operational products (IMDAA/MERA reanalysis, IMD gridded station observations). Ingesting IMDAA/MERA directly is the natural next step for a fully NCMRWF-native pipeline; the architecture's pluggable ingestion clients (`ingestion/clients.py`) are designed to make this a contained change.


### 1. Multi-model forecast (physical NWP + ensemble)
Endpoint: `https://api.open-meteo.com/v1/forecast`
Key params:
- `latitude`, `longitude`
- `hourly` = comma-separated variables, e.g. `temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m`
- `models` = comma-separated model list, e.g. `gfs_seamless,icon_seamless,ecmwf_ifs025`
- `forecast_days` = up to 16

### 2. AI model forecast (ECMWF AIFS)
Endpoint: ECMWF-specific Open-Meteo endpoint (check current docs at
https://open-meteo.com/en/docs/ecmwf-api — model IDs there include `ecmwf_ifs`, `ecmwf_ifs025`,
and `ecmwf_aifs025_single`).
- AIFS is ECMWF's GNN-based AI weather model — this is what makes the system genuinely
  "hybrid AI-NWP" rather than NWP-only.
- Note: AIFS currently only provides 6-hourly time steps (coarser than IFS). Handle this
  resolution mismatch explicitly in ingestion (interpolate or bucket to matching lead times) —
  do not silently compare 1-hourly IFS to 6-hourly AIFS without reconciling timesteps.

### 3. Ensemble forecast
Endpoint: Open-Meteo Ensemble API (see https://open-meteo.com/en/docs/ensemble-api) — provides
spread across ensemble members (ECMWF, GFS, ICON, GEM, UKMO ensembles). Use ensemble spread as a
confidence/uncertainty signal for the extreme weather module.

### 4. Historical / archive data (ground truth for training + verification)
Endpoint: `https://archive-api.open-meteo.com/v1/archive` (historical weather API, decades of
reanalysis-based data). Use this as your "observed" ground truth for:
- computing each model's historical skill (Module 2)
- the leak-free train/calibrate/test time split (Module 5)

## Variables to pull (minimum set)
- `temperature_2m`
- `precipitation`
- `wind_speed_10m`, `wind_gusts_10m`

## Target locations (lat/lon — fill in exact coordinates in ingestion config)
| City | Zone |
|---|---|
| Mumbai | Coastal |
| Delhi | Semi-arid |
| Chennai | Coastal/monsoon |
| Kolkata | Deltaic/cyclone-exposed |
| Guwahati | High-rainfall NE |
| Jaisalmer | Arid |
| Shimla | Hill/orographic |
| Bhubaneswar | Cyclone-prone east coast |
| Bengaluru | Plateau |
| Thiruvananthapuram | Coastal monsoon |

## Extreme weather thresholds
Do NOT hardcode assumed IMD thresholds without verification. Before finalizing Module 4:
1. Look up current IMD operational definitions for heavy rainfall, heatwave, and high-wind
   warnings (these vary by region — e.g. heatwave criteria differ for plains vs. coastal areas).
2. Cite the source in a code comment next to the threshold constant.
3. If unverified at build time, mark clearly as `# PLACEHOLDER` so it's not mistaken for a
   verified operational threshold during judging.

## Rate limits / practical notes
- Open-Meteo free tier is generous but not unlimited — cache responses locally during
  development so you're not re-fetching identical data on every test run.
- Build ingestion to tolerate a single source being temporarily unavailable (see
  ARCHITECTURE.md Module 1).
