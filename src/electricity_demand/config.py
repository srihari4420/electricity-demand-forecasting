"""Project settings. Paths are worked out relative to the repo root so the whole
thing runs from a fresh clone with no hard-coded directories."""

from pathlib import Path

# repo root is two levels up from this file (src/electricity_demand/config.py)
ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"

OUT_FIGURES = ROOT / "outputs" / "figures"
OUT_FORECASTS = ROOT / "outputs" / "forecasts"
OUT_METRICS = ROOT / "outputs" / "metrics"
OUT_MODELS = ROOT / "outputs" / "model_objects"

# data sources
LOAD_URL = ROOT / "data" / "raw" / "time_series_60min_singleindex.csv"
LOAD_COL = "DE_load_actual_entsoe_transparency"     # ENTSO-E actual total load, Germany
TEMPERATURE_URL = "https://archive-api.open-meteo.com/v1/archive"

# target
TARGET = "load_gw"          # weekly average load, converted from MW to GW
START = "2015-01-01"

# forecasting setup
SEASON = 52                 # weekly data, annual cycle
HORIZON_WEEKS = 104         # final two years held out for testing
RANDOM_SEED = 42

# Berlin stands in for German temperature
BERLIN_LAT, BERLIN_LON = 52.52, 13.41
HEATING_BASE_C = 15.5       # heating degree-day base
COOLING_BASE_C = 22.0       # cooling degree-day base

# default SARIMAX orders (README starting point; the grid can refine them)
SARIMAX_ORDER = (1, 1, 1)
SARIMAX_SEASONAL_ORDER = (1, 1, 1, SEASON)


def ensure_dirs():
    """Create the output folders if they are not already there."""
    for d in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED,
              OUT_FIGURES, OUT_FORECASTS, OUT_METRICS, OUT_MODELS):
        d.mkdir(parents=True, exist_ok=True)
