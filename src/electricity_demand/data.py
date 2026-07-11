"""Getting the data in and into shape. Load comes from Open Power System Data,
temperature from the Open-Meteo archive. Everything downstream works on the
weekly average load in GW."""

import pandas as pd
import requests

from . import config


def download_load(url=config.LOAD_URL, col=config.LOAD_COL, start=config.START):
    """Read the OPSD 60-minute file straight from the URL and keep all-Germany
    actual load from 2015 onwards. Returns an hourly series in MW."""
    df = pd.read_csv(url, index_col=0, parse_dates=True)
    s = df[col].loc[start:].copy()
    s.name = "load_mw"
    return s


def clean_hourly(hourly):
    """Put the series on a regular hourly grid, drop the timezone (UTC stamps,
    but at a weekly mean the offset against Berlin time does not matter), and
    fill the few short gaps by time interpolation."""
    hourly = hourly.astype(float)
    if hourly.index.tz is not None:
        hourly.index = hourly.index.tz_localize(None)
    regular = hourly.asfreq("h")
    n_gaps = int(regular.isna().sum())
    filled = regular.interpolate("time")
    filled.name = "load_mw"
    return filled, n_gaps


def to_weekly_gw(hourly):
    """Weekly average load, converted from MW to GW. This is the target."""
    weekly = hourly.resample("W").mean() / 1000.0
    weekly.name = config.TARGET
    return weekly


def download_temperature(lat=config.BERLIN_LAT, lon=config.BERLIN_LON,
                         start=config.START, end="2020-12-31"):
    """Daily mean temperature for Berlin from the Open-Meteo archive."""
    params = {"latitude": lat, "longitude": lon, "start_date": start,
              "end_date": end, "daily": "temperature_2m_mean",
              "timezone": "Europe/Berlin"}
    r = requests.get(config.TEMPERATURE_URL, params=params, timeout=60)
    r.raise_for_status()
    d = r.json()["daily"]
    out = pd.DataFrame({"temperature_2m_mean": d["temperature_2m_mean"]},
                       index=pd.to_datetime(d["time"]))
    out.index.name = "date"
    return out


def build_weekly_load(save=True):
    """Full load path: download, clean, aggregate to weekly GW. Optionally cache
    the raw hourly and the processed weekly series under data/."""
    config.ensure_dirs()
    hourly_raw = download_load()
    hourly, n_gaps = clean_hourly(hourly_raw)
    weekly = to_weekly_gw(hourly)
    if save:
        hourly.to_frame().to_parquet(config.DATA_RAW / "load_hourly_mw.parquet")
        weekly.to_frame().to_parquet(config.DATA_PROCESSED / "load_weekly_gw.parquet")
    return weekly, hourly, n_gaps
