# PRD — Hybrid AI–NWP Multi-Model Forecast Blending System
SIH Problem Statement 26081 | Ministry of Earth Sciences / NCMRWF | Theme: Disaster Management

## 1. Problem
Different forecast sources (physical NWP models, ensemble systems, AI/ML weather models) have
different strengths depending on region, season, lead time, and weather regime. No single source
is best everywhere. Naive averaging of sources smooths out extreme values (heavy rainfall,
heatwaves, high wind), which is dangerous for disaster management use cases.

## 2. Goal
Build a system that dynamically blends multiple live forecast sources (including at least one
true AI/ML model, not just physical NWP) using adaptive, learned weights, and produces:
rainfall, temperature, and wind forecasts, plus extreme-weather guidance, that are demonstrably
more skillful than any single source or a naive average — verified honestly, not just asserted.

## 3. Required deliverables (from the PS — do not drop any of these)
1. Dynamically blended forecast (rainfall, temperature, wind)
2. Model weight maps (which source is trusted where, and why)
3. Demonstrated improved forecast skill vs. individual models (quantified)
4. Extreme weather guidance (heavy rainfall / heatwave / high wind)
5. Operational workflow — an automated, repeatable pipeline, not a one-off script

## 4. Differentiators (why this beats other public SIH 26081 submissions)
- **Live data at demo time.** Pulls real forecasts from Open-Meteo (IFS, AIFS, GFS, ICON, ENS)
  rather than synthetic/mock data or an offline-only "demo mode."
- **Leak-free, year-split verification, reported honestly.** Train / calibrate / test periods are
  strictly separated by time. Skill scores (RMSE, POD, FAR, CSI, ETS) are reported including
  cases where the system underperforms. No cherry-picked wins only.
- **Quantitative extreme-event case study.** At least one real historical extreme event is
  backtested, showing naive averaging underestimating it and the blended system doing measurably
  better (with numbers on screen, not just a claim).
- **District-level, plain-language alerts** in at least one regional language, across at least
  two climate zones.

## 5. Non-goals (explicitly out of scope for the hackathon build)
- No claim of matching operational-grade NCMRWF forecast accuracy.
- No invented IMD threshold numbers — thresholds used for extreme detection must be sourced and
  cited in code comments, or clearly labeled as approximate/placeholder pending verification.
- No fabricated or hallucinated verification results — every skill score shown must come from an
  actual computed backtest, never a hardcoded/placeholder number.

## 6. Target locations (climate-zone coverage for weight-map contrast)
Mumbai (coastal), Delhi (semi-arid), Chennai (coastal/monsoon), Kolkata (deltaic/cyclone-exposed),
Guwahati (high rainfall NE), Jaisalmer (arid), Shimla (hill/orographic), Bhubaneswar
(cyclone-prone east coast), Bengaluru (plateau), Thiruvananthapuram (coastal monsoon).

## 7. Success criteria for the hackathon demo
- Live pipeline runs end-to-end in front of judges (ingestion → blend → dashboard) without
  manual data injection.
- Weight map visibly differs by region/season in a way that's explainable in one sentence per
  region.
- At least one real extreme event is shown with before/after numbers (naive avg vs. blend).
- Verification panel shows real computed skill scores, including at least one honest limitation.
- District alert output is legible to a non-technical end user in a regional language.
