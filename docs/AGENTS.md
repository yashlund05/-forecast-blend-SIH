# Agent Instructions

You are building the Hybrid AI–NWP Multi-Model Forecast Blending System for SIH PS 26081.
Read PRD.md, ARCHITECTURE.md, DATA_SOURCES.md, TECH_STACK.md, and TASKS.md before writing code.
Work through TASKS.md in phase order and check items off as completed.

## Hard rules — do not violate these even if they'd make a demo look better faster

1. **Never fabricate data or results.** Do not hardcode a "skill score," a "verification metric,"
   or a "case study" number that wasn't actually computed by running the code against real data.
   If a real computation isn't ready yet, leave the UI element clearly showing "not yet computed"
   rather than a plausible-looking fake number. Judges may ask to see the code path that produced
   any number on screen.

2. **Never present synthetic/mock data as if it were live.** If a data source is temporarily
   unavailable and you fall back to cached or synthetic data for development, label it visibly
   (e.g. a small "using cached data" badge) — do not silently swap it in without indication.

3. **Do not invent IMD threshold values.** Extreme-weather thresholds (heavy rainfall, heatwave,
   high wind) must either be sourced with a citation in a code comment, or explicitly marked
   `# PLACEHOLDER - needs verification` in the code and flagged in any UI text that references
   them.

4. **Keep the train/calibrate/test time split strictly non-overlapping.** This is the single most
   important credibility factor for the verification module — a leaked split invalidates the
   whole skill-score story. Add an assertion or test that fails loudly if the date ranges overlap.

5. **Report failure cases, not just wins.** The verification dashboard must show at least one
   scenario where the blended system does not outperform individual sources. Do not filter these
   out to make the results look cleaner.

6. **Handle per-source API failures gracefully.** One weather source being down must degrade the
   system (fewer sources in the blend, logged clearly) rather than crash the pipeline.

7. **No hardcoded credentials.** Open-Meteo needs no API key; if any other service is added later
   that does need one, use environment variables / a `.env` file, never a literal key in code.

## Style/conventions
- Python: follow PEP8, type hints on function signatures, docstrings on public functions.
- Keep modules matching the folder structure in ARCHITECTURE.md — don't collapse everything into
  one script.
- Write a unit test for any function that computes a number displayed on the dashboard
  (weights, skill scores, extreme thresholds) — these are the numbers judges will question.
- Commit incrementally per phase/task, with descriptive commit messages — a visible build history
  is part of the credibility story for this project.

## When uncertain
If a requirement in PRD.md/TASKS.md is ambiguous or you're not sure a claimed data source/API
detail is still accurate (endpoints and model names can change), verify against current
Open-Meteo documentation rather than assuming the details in DATA_SOURCES.md are still exact —
treat that file as a strong starting pointer, not a guaranteed-current API reference.
