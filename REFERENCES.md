# References

This document catalogs every external source, API, dataset, domain standard, and documentation page explicitly cited or relied upon within the repository's codebase and documentation. 

Per project rules, no fabricated DOIs, years, or authors are included. Items that lack formal citations in the repository are explicitly flagged.

## Data Sources & APIs

**Open-Meteo**
*   Open-Meteo. (n.d.). *Base Documentation*. URL: https://open-meteo.com
*   Open-Meteo. (n.d.). *Multi-Model Forecast API*. URL: https://api.open-meteo.com/v1/forecast
*   Open-Meteo. (n.d.). *Historical Archive API (ERA5-backed reanalysis)*. URL: https://archive-api.open-meteo.com/v1/archive
*   Open-Meteo. (n.d.). *Historical Forecast API*. URL: https://historical-forecast-api.open-meteo.com/v1/forecast
*   Open-Meteo. (n.d.). *Ensemble API*. URL: https://ensemble-api.open-meteo.com/v1/ensemble
*   Open-Meteo. (n.d.). *Ensemble API Documentation*. URL: https://open-meteo.com/en/docs/ensemble-api
*   Open-Meteo. (n.d.). *ECMWF Models API (AIFS)*. URL: https://api.open-meteo.com/v1/ecmwf

**NCMRWF**
*   NCMRWF. (n.d.). *NCUM Operational Information (IMDAA/MERA Access)*. URL: https://ncmrwf.gov.in/ncum_op_info.html

## AI/ML Model Documentation

*Note: The repository integrates these AI models via Open-Meteo endpoints but does not formally cite their underlying academic papers or technical documentation URLs.*

*   **ECMWF AIFS (Artificial Intelligence Forecasting System)**: Referenced in `DATA_SOURCES.md` as a Graph Neural Network (GNN) developed by ECMWF, but no external paper or official documentation URL is cited.
*   **Google DeepMind WeatherNext 2**: Referenced in `DATA_SOURCES.md` as a 64-member global AI ensemble, but no external paper or official documentation URL is cited.

## Domain Standards (IMD Thresholds)

### Verified Citations

*   **India Meteorological Department. (2021, March).** *Standard Operation Procedure - Weather Forecasting and Warning Services*. URL: https://mausam.imd.gov.in/imd_latest/contents/pdf/forecasting_sop.pdf
    *   **Rainfall / Cloudburst**: Cited for 24-hour accumulated rainfall intensity (Chapter 1, Section 1.7.2, Table 1.5, p. 10) and Cloudburst criteria (Chapter 5, Section 5.3, p. 96).
    *   **Heatwave Base Thresholds**: Cited for topography-specific temperature cutoffs (Chapter 7, Section 7.3.1, p. 160).
    *   **Squall (Convective/Thunderstorm)**: Cited for surface wind gust classifications (Chapter 6, Section 6.3.1, p. 140).
    *   **Gale (Synoptic/Cyclone)**: Cited for maritime cyclonic wind classifications (Chapter 10, Section 10.3.1, Table 10.7, p. 249; Chapter 12).

### Unverified / Placeholder Standards

*The following criteria are explicitly marked as `# PLACEHOLDER` or `# UNVERIFIED` in the repository code (`extremes/thresholds.py` and `blending/regime.py`) and require verification against official sources before operational use:*

*   **India Meteorological Department. (2021).** *Forecaster's Guidelines on Monsoon*. Chapter 3: Active/Break Monsoon Criteria. 
    *   *Status*: Flagged as a `# PLACEHOLDER` proxy (0.3 mm/hr rolling 7-day mean).
*   **NDMA Heat Action Plans**: 
    *   *Status*: Flagged as `# UNVERIFIED` operational proxies for intermediate regional heatwave tiers (e.g., orange alerts for deltaic/coastal classes).
*   **Hourly Heavy Rain Spells**: 
    *   *Status*: Proxies for intense hourly spells (5.0 mm, 15.0 mm, 30.0 mm) are flagged as `# UNVERIFIED`.

## Methodologies & Software/Libraries

### Methodologies

*   **Inverse-Error Weighting & Inverse Distance Weighting (IDW)**: 
    *   *Status*: No academic papers (e.g., Shepard 1968) or technical writeups are cited for the weighting arithmetic or spatial interpolation logic.

### Libraries

*   **Python Stack**: 
    *   *Status*: Libraries such as `pandas`, `numpy`, `scikit-learn`, `streamlit`, `plotly`, `requests`, `httpx`, `schedule`, and `pytest` are listed in `requirements.txt` and `TECH_STACK.md`, but no formal academic DOIs, whitepapers, or documentation URLs are explicitly cited in the codebase or documentation.
