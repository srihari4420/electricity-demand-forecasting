"""Feature engineering. Two rules run through everything here: preprocessing
that learns from the data is fitted on train only (done in the models), and lag
and rolling features look strictly backwards so nothing leaks across the
forecast origin."""

import numpy as np
import pandas as pd

from . import config

try:
    import holidays as holidays_lib
    _HAS_HOLIDAYS = True
except ImportError:                     # the pipeline still runs without holiday features
    _HAS_HOLIDAYS = False


def weekly_temperature_features(temp_daily, weekly_index,
                                base_heat=config.HEATING_BASE_C,
                                base_cool=config.COOLING_BASE_C):
    """Roll daily Berlin temperature up to the weekly load grid. Heating and
    cooling degree-days isolate the cold and hot extremes, which is where
    temperature really moves load."""
    daily = temp_daily["temperature_2m_mean"]
    agg = pd.DataFrame({
        "temp_mean": daily.resample("W").mean(),
        "temp_min": daily.resample("W").min(),
        "temp_max": daily.resample("W").max(),
        "heating_degree_days": np.maximum(base_heat - daily, 0).resample("W").sum(),
        "cooling_degree_days": np.maximum(daily - base_cool, 0).resample("W").sum(),
    })
    tw = agg.reindex(weekly_index)          # align by date, then fill any edge weeks
    return tw.interpolate("linear").bfill().ffill()


def holiday_features(weekly_index, country="DE"):
    """Count public holidays in each week. Holidays are known in advance, so
    unlike temperature these are valid future covariates. Falls back to zeros if
    the holidays package is not installed."""
    out = pd.DataFrame(index=weekly_index)
    if not _HAS_HOLIDAYS:
        out["holiday_days"] = 0
        out["has_holiday"] = 0
        return out
    years = range(weekly_index.min().year, weekly_index.max().year + 1)
    de = holidays_lib.country_holidays(country, years=list(years))
    # a week is labelled by its end date; count holidays in the 7 days up to it
    counts = []
    for wk_end in weekly_index:
        days = pd.date_range(wk_end - pd.Timedelta(days=6), wk_end, freq="D")
        counts.append(sum(d.date() in de for d in days))
    out["holiday_days"] = counts
    out["has_holiday"] = (out["holiday_days"] > 0).astype(int)
    return out


def calendar_features(index, season=config.SEASON):
    """Week-of-year, month, and a few Fourier pairs that give a smooth handle on
    the annual cycle."""
    week = index.isocalendar().week.astype(int).to_numpy()
    out = pd.DataFrame(index=index)
    out["week"] = week
    out["month"] = index.month
    for k in range(1, 4):
        out[f"sin_{k}"] = np.sin(2 * np.pi * k * week / season)
        out[f"cos_{k}"] = np.cos(2 * np.pi * k * week / season)
    return out


def lag_features(target, lags=(1, 2, 4, 8, 13, 26, 52),
                 roll_windows=(4, 13, 52)):
    """Lagged target plus rolling means. The shift(1) before each rolling mean
    keeps the current week out of its own predictors."""
    out = pd.DataFrame(index=target.index)
    for lag in lags:
        out[f"lag_{lag}"] = target.shift(lag)
    for w in roll_windows:
        out[f"roll_mean_{w}"] = target.shift(1).rolling(w).mean()
    return out


def build_feature_table(weekly_load, temp_daily=None, add_holidays=True):
    """Assemble the supervised table: target, calendar, holidays, temperature and
    lag/rolling features. Rows with any missing feature (the early weeks the lags
    cannot reach) are dropped."""
    df = pd.DataFrame({config.TARGET: weekly_load})
    df = df.join(lag_features(weekly_load))
    df = df.join(calendar_features(weekly_load.index))
    if add_holidays:
        df = df.join(holiday_features(weekly_load.index))
    if temp_daily is not None:
        temp = weekly_temperature_features(temp_daily, weekly_load.index)
        temp["heating_degree_days_lag1"] = temp["heating_degree_days"].shift(1)
        temp["temp_mean_lag1"] = temp["temp_mean"].shift(1)
        df = df.join(temp)
    return df.dropna()
