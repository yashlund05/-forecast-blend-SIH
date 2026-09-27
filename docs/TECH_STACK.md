# Tech Stack

## Backend / pipeline
- **Language**: Python 3.11+
- **Data handling**: pandas, numpy
- **HTTP/API calls**: requests (or httpx for async ingestion of multiple sources in parallel)
- **ML — baseline weighting**: plain numpy/pandas (inverse-error weighting is not a "model," just
  arithmetic — keep it dependency-light)
- **ML — stretch gating model**: scikit-learn or LightGBM for regime-aware weight prediction
- **Scheduling**: `schedule` library or a simple cron entry calling a `run_pipeline.py` script —
  do not over-engineer with a full task queue for a hackathon build
- **Storage**: SQLite (via sqlite3 or SQLAlchemy) for simplicity, or flat parquet/CSV files if
  the team prefers — either is fine, just be consistent

## Verification
- scikit-learn metrics (MAE, RMSE) for continuous variables
- Custom implementations for POD, FAR, CSI, ETS (contingency-table-based extreme event metrics —
  these are simple to implement directly, no special library needed)

## Frontend / dashboard
- **Recommended**: Streamlit — fastest path from Python backend to an interactive dashboard,
  minimal separate frontend build needed
- **Alternative if team has frontend strength**: React + a mapping library (Leaflet or
  Mapbox GL) + a charting library (Recharts or Plotly) + FastAPI backend serving JSON to it
- **Map component**: Leaflet (free, no API key) or Plotly's choropleth/scatter-geo for India

## Alerts
- Templated string generation (Python f-strings or Jinja2) with a small dictionary of
  region-language strings — no need for a full translation API/model for the hackathon scope

## Dev tooling
- `pytest` for unit tests (especially: verification time-split enforcement, threshold logic,
  weight computation correctness)
- `.env` file for any config (though Open-Meteo needs no API key, keep this pattern ready in
  case you add a source that does)
- Git repo with clear commit history — judges may look at this to gauge genuine incremental
  build vs. last-minute dump

## What NOT to add (keep scope hackathon-realistic)
- No heavy deep learning training pipeline (spatial U-Nets, custom GNNs) unless the team already
  has strong ML members with time to spare — several competitor repos claim this but it adds
  large risk for uncertain payoff in a time-boxed build. Prioritize the verification and
  extreme-event proof over a fancier model architecture.
- No Kubernetes/microservices — a single well-organized Python project is more defensible in a
  judging Q&A than an over-engineered stack you can't explain end-to-end.
