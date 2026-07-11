"""Figures. Every function returns a Matplotlib axes so a notebook can show it
and a script can save it with `save`."""

import numpy as np
import matplotlib.pyplot as plt   # backend auto-selects: inline in notebooks, Agg headless

from statsmodels.graphics.tsaplots import plot_acf


def save(ax, path, dpi=130):
    fig = ax.get_figure()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def forecast_comparison(train, test, forecasts, title="Forecast comparison", tail=104):
    """Actual test period with every model's forecast overlaid."""
    fig, ax = plt.subplots(figsize=(13, 6))
    train.iloc[-tail:].plot(ax=ax, color="black", lw=1, label="train")
    test.plot(ax=ax, color="steelblue", lw=2.2, label="actual")
    for name, fc in forecasts.items():
        fc.reindex(test.index).plot(ax=ax, lw=1.1, ls="--", label=name)
    ax.set_ylabel("Load (GW)")
    ax.set_title(title)
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()
    return ax


def single_forecast(train, test, mean_fc, ci=None, title="", tail=78):
    fig, ax = plt.subplots(figsize=(13, 5))
    train.iloc[-tail:].plot(ax=ax, color="black", lw=1, label="train")
    test.plot(ax=ax, color="steelblue", lw=1.6, label="actual")
    mean_fc.plot(ax=ax, color="firebrick", lw=1.4, label="forecast")
    if ci is not None:
        ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1],
                        color="firebrick", alpha=0.15, label="95% interval")
    ax.set_ylabel("Load (GW)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return ax


def error_diagnostics(test, forecasts, title="Forecast errors"):
    """Errors over the test period for each model, plus their spread."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for name, fc in forecasts.items():
        err = (fc.reindex(test.index) - test)
        err.plot(ax=ax[0], lw=1, label=name)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_title("Forecast error over time")
    ax[0].set_ylabel("Error (GW)")
    ax[0].legend(fontsize=8, ncol=2)
    data = [(fc.reindex(test.index) - test).dropna().values for fc in forecasts.values()]
    ax[1].boxplot(data, tick_labels=list(forecasts.keys()))
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_title("Error distribution")
    ax[1].set_ylabel("Error (GW)")
    ax[1].tick_params(axis="x", rotation=90)
    fig.suptitle(title)
    fig.tight_layout()
    return ax[0]


def residual_acf(resid, lags=52, title="Residual ACF"):
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(resid.dropna(), lags=lags, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return ax


def prediction_intervals(test, mean_fc, lower, upper, title="Prediction intervals"):
    fig, ax = plt.subplots(figsize=(13, 5))
    test.plot(ax=ax, color="steelblue", lw=1.6, label="actual")
    mean_fc.plot(ax=ax, color="firebrick", lw=1.4, label="forecast")
    ax.fill_between(mean_fc.index, lower, upper, color="firebrick", alpha=0.15,
                    label="95% interval")
    ax.set_ylabel("Load (GW)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return ax


def series_overview(daily_gw, weekly_gw):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    daily_gw.plot(ax=ax[0], lw=0.7)
    ax[0].set_title("German load, daily average")
    ax[0].set_ylabel("Load (GW)")
    weekly_gw.plot(ax=ax[1], lw=1)
    ax[1].set_title("German load, weekly average")
    ax[1].set_ylabel("Load (GW)")
    fig.tight_layout()
    return ax[0]
