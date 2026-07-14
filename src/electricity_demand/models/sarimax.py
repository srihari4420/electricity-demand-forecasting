import warnings
import itertools

import pandas as pd
import statsmodels.api as sm

from .. import config


def fit_sarimax(train, order=config.SARIMAX_ORDER,
                seasonal_order=config.SARIMAX_SEASONAL_ORDER, exog=None):
    return sm.tsa.statespace.SARIMAX(
        train, exog=exog, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)


def forecast(model, steps, exog=None, index=None, alpha=0.05):
    """Return the mean forecast and the confidence band, both on `index`."""
    fc = model.get_forecast(steps=steps, exog=exog)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=alpha)
    if index is not None:
        mean.index = index
        ci.index = index
    return mean, ci


def grid_search(train, seasonal_order=config.SARIMAX_SEASONAL_ORDER,
                p_range=range(0, 4), d_range=range(0, 3), q_range=range(0, 4),
                exog=None, verbose=False):
    """Sweep the non-seasonal order at a fixed seasonal order, ranked by AIC.
    Ranges are modest by default; widen to p=0..6, d=0..2, q=0..6 for the full
    grid, which is slow at season 52."""
    records = []
    combos = list(itertools.product(p_range, d_range, q_range))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, (p, d, q) in enumerate(combos, 1):
            try:
                res = fit_sarimax(train, (p, d, q), seasonal_order, exog=exog)
                records.append({"order": (p, d, q), "aic": res.aic})
                tag = f"AIC={res.aic:.1f}"
            except Exception:
                tag = "failed"
            if verbose:
                print(f"[{i:3d}/{len(combos)}] {(p, d, q)} {tag}")
    return pd.DataFrame(records).sort_values("aic").reset_index(drop=True)
