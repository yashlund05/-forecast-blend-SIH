"""Provisional draft thresholds for extreme weather contingency evaluation.

# PLACEHOLDER - pending Phase 5 IMD source verification
These threshold constants represent commonly-cited meteorological alert criteria
used to establish verification math (POD, FAR, CSI, ETS) in Phase 4.
Per AGENTS.md Hard Rule 3, these are explicitly flagged as placeholders
and will be replaced with verified regional IMD definitions in Phase 5.
"""

# PLACEHOLDER - pending Phase 5 IMD source verification
# IMD criteria for heavy rainfall: >= 64.5 mm in 24 hours.
# Hourly proxy burst threshold: >= 5.0 mm/hr
HEAVY_RAIN_HOURLY_THRESHOLD_MM = 5.0  # PLACEHOLDER - pending Phase 5 IMD source verification
HEAVY_RAIN_24H_THRESHOLD_MM = 64.5   # PLACEHOLDER - pending Phase 5 IMD source verification

# PLACEHOLDER - pending Phase 5 IMD source verification
# Heatwave threshold proxy: max surface temperature >= 40.0°C (for plains)
HEATWAVE_THRESHOLD_TEMP_C = 40.0     # PLACEHOLDER - pending Phase 5 IMD source verification

# PLACEHOLDER - pending Phase 5 IMD source verification
# High wind / squall speed proxy: sustained wind speed >= 40.0 km/h
HIGH_WIND_THRESHOLD_KMH = 40.0       # PLACEHOLDER - pending Phase 5 IMD source verification
