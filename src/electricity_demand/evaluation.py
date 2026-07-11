"""Forecast metrics. The four the brief asks for are MAE, RMSE, MASE and bias,
all computed the same way for every model so the comparison is fair."""

import numpy as np
import pandas as pd

from . import config


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def bias(y_true, y_pred):
    """Mean(pred - actual). Positive means the model runs high."""
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def mase(y_true, y_pred, y_train, season=config.SEASON):
    """MAE scaled by the in-sample seasonal-naive error. Under 1 beats a
    seasonal-naive guess. A perfect forecast gives 0."""
    y_train = np.asarray(y_train)
    denom = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    return mae(y_true, y_pred) / denom


def evaluate(y_true, y_pred, y_train, name, season=config.SEASON):
    """One metric row for a model."""
    return {
        "model": name,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MASE": mase(y_true, y_pred, y_train, season),
        "Bias": bias(y_true, y_pred),
    }


def metrics_table(rows, baseline="seasonal_naive"):
    """Collect model rows into a table, sorted by RMSE, with the gap to the
    seasonal-naive benchmark."""
    df = pd.DataFrame(rows).set_index("model").sort_values("RMSE")
    if baseline in df.index:
        base = df.loc[baseline, "RMSE"]
        df["vs_snaive_%"] = (df["RMSE"] / base - 1) * 100
    return df.round(3)
