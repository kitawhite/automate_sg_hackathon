"""Central configuration. Everything is keyed on airport code so the pipeline
is generic; to add an airport, extend RUNWAY_HEADING (+ STATION_COORDS)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "sources" / "Flight_Data_With_Weather_Final.csv"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# --- Focus airports and their (assumed) runway headings in degrees. -----------
# The brief fixes ATL ops to heading 080 and DFW to 180. The pipeline accepts
# any ICAO/IATA code present in the data; these two are what we validate on.
AIRPORTS = ["ATL", "DFW"]
RUNWAY_HEADING = {"ATL": 80.0, "DFW": 180.0}

# Station coords (lat, lon, alt_m) + local timezone, for the wind-direction
# enrichment. enrich.py fetches REAL hourly wind direction/speed/gusts from the
# Open-Meteo historical archive (free, no key); falls back to meteostat, then to
# a synthetic wdir if offline. Timezone must match the airport's local clock so
# the hourly series aligns with the (local) flight timestamps.
STATION_COORDS = {
    "ATL": (33.6367, -84.4281, 313.0),
    "DFW": (32.8998, -97.0403, 185.0),
}
STATION_TZ = {
    "ATL": "America/New_York",
    "DFW": "America/Chicago",
}

# --- Forecasting horizon ------------------------------------------------------
# The challenge asks for 3-12h-ahead forecasts. We expose several leads; every
# feature is the weather KNOWN at (target_hour - lead), which is the leakage guard.
LEADS = [3, 6, 12]
MIN_LEAD = min(LEADS)

# AAR aggregation bucket. '1h' per the brief. NOTE: the provided file is a SAMPLE
# of flights, so hourly arrival counts are small; switch to '3h' here if the
# hourly signal is too sparse for your taste (documented in README).
BUCKET = "1h"

# Severe-delay threshold (minutes of arrival delay).
SEVERE_DELAY_MIN = 45.0

# Strict temporal split: everything before TEST_START trains; the rest tests.
TEST_START = "2025-01-01"

# Weather columns present in the CSV, per role suffix (_ORIGIN / _DEST).
WEATHER_VARS = ["temp", "prcp", "wspd", "wpgt", "pres", "snow", "coco"]

RANDOM_SEED = 42
