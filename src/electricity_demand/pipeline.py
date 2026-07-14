"""The reusable pipeline. `run_pipeline` walks the whole workflow, from loading
the data to saving forecasts, metrics and figures, and returns the results so a
notebook or script can pick them up. The heavy lifting lives in the model
modules; this file just wires them together and keeps the train-test discipline
consistent across models."""

import numpy as np
import pandas as pd

from . import config, data as data_mod, features as feat, evaluation as ev, plotting as viz
from .models import (BenchmarkForecaster, fit_sarimax, sarimax_forecast,
                     fit_predict, BayesianForecaster, neural)

# the model that stands in for the "feature_model" column, chosen up front rather
# than by peeking at the test set
FEATURE_MODEL = "random_forest"


def _split(series, horizon=config.HORIZON_WEEKS):
    return series.iloc[:-horizon], series.iloc[-horizon:]


def run_pipeline(weekly=None, temp_daily=None, save=True, run_grid=False,
                 with_bayesian=True, with_neural=None, verbose=True):
    """Run the full comparison. Pass `weekly` and `temp_daily` to reuse
    already-loaded data (handy for tests); leave them as None to download.
    `with_neural` defaults to True when TensorFlow is available."""
    if save:
        config.ensure_dirs()
    if with_neural is None:
        with_neural = neural.available()

    def log(msg):
        if verbose:
            print(msg)

    # --- steps 1-3: load, clean, aggregate to weekly GW ------------------------
    if weekly is None:
        log("Downloading and preparing load data ...")
        weekly, hourly, n_gaps = data_mod.build_weekly_load(save=save)
        log(f"  weekly points: {len(weekly)} | hourly gaps filled: {n_gaps}")
    if temp_daily is None:
        log("Downloading Berlin temperature ...")
        temp_daily = data_mod.download_temperature(
            start=str(weekly.index.min().date()), end=str(weekly.index.max().date()))

    # --- step 4: features ------------------------------------------------------
    log("Building features ...")
    table = feat.build_feature_table(weekly, temp_daily=temp_daily, add_holidays=True)
    # exogenous frame on the full weekly grid for SARIMAX
    exog = feat.weekly_temperature_features(temp_daily, weekly.index)
    exog = exog.join(feat.holiday_features(weekly.index))
    exog_cols = ["heating_degree_days", "cooling_degree_days", "holiday_days"]

    # --- step 5: train / test split (chronological, last 104 weeks) ------------
    train, test = _split(weekly)
    test_index = test.index
    exog_train, exog_test = exog.loc[train.index, exog_cols], exog.loc[test_index, exog_cols]

    forecasts = {}      # name -> Series on the test index
    intervals = {}      # name -> (lower, upper) for models that provide them

    # --- step 6: benchmarks ----------------------------------------------------
    log("Fitting benchmarks ...")
    forecasts.update(BenchmarkForecaster(train).all(test_index))

    # --- step 7: SARIMAX (conditional on observed temperature) -----------------
    log("Fitting SARIMAX ...")
    sarimax_model = fit_sarimax(train, exog=exog_train)
    sarimax_mean, sarimax_ci = sarimax_forecast(
        sarimax_model, steps=len(test_index), exog=exog_test, index=test_index)
    forecasts["sarimax"] = sarimax_mean.rename("sarimax")
    intervals["sarimax"] = (sarimax_ci.iloc[:, 0], sarimax_ci.iloc[:, 1])

    # --- step 8: feature-based models ------------------------------------------
    log("Fitting feature-based models ...")
    feat_train = table.loc[table.index < test_index[0]]
    feat_test = table.loc[test_index]
    X_train, y_train_f = feat_train.drop(columns=config.TARGET), feat_train[config.TARGET]
    X_test = feat_test.drop(columns=config.TARGET)
    feat_preds, feat_fitted = fit_predict(X_train, y_train_f, X_test)
    for name, pred in feat_preds.items():           # keep each for the metrics table
        forecasts[name] = pred
    forecasts["feature_model"] = feat_preds[FEATURE_MODEL].rename("feature_model")

    # --- step 9: Bayesian and neural (optional) --------------------------------
    if with_bayesian:
        log("Fitting Bayesian regression ...")
        bayes = BayesianForecaster().fit(X_train, y_train_f)
        b_mean, b_lo, b_hi = bayes.predict(X_test)
        forecasts["bayesian"] = b_mean
        intervals["bayesian"] = (b_lo, b_hi)
    if with_neural:
        log("Fitting neural (weekly LSTM) ...")
        try:
            model = neural.WeeklyLSTM().fit(train)
            forecasts["neural"] = model.forecast(test_index, weekly)
        except Exception as exc:
            log(f"  neural model skipped: {exc}")

    # --- steps 10-11: assemble forecasts and evaluate --------------------------
    log("Evaluating ...")
    all_fc = pd.DataFrame({"actual": test})
    for name, fc in forecasts.items():
        all_fc[name] = fc.reindex(test_index)

    rows = [ev.evaluate(test.values, all_fc[name].values, train.values, name)
            for name in forecasts if all_fc[name].notna().all()]
    metrics = ev.metrics_table(rows)

    # --- step 12: save outputs -------------------------------------------------
    if save:
        # tidy column order for the headline forecast file
        headline = ["actual", "mean", "naive", "seasonal_naive", "drift",
                    "sarimax", "feature_model", "bayesian", "neural"]
        cols = [c for c in headline if c in all_fc.columns]
        all_fc[cols].to_csv(config.OUT_FORECASTS / "all_forecasts.csv")
        all_fc.to_csv(config.OUT_FORECASTS / "all_forecasts_full.csv")
        metrics.reset_index().to_csv(config.OUT_METRICS / "model_comparison.csv", index=False)

        headline_fc = {k: forecasts[k] for k in
                       ["seasonal_naive", "sarimax", "linear_regression",
                        "random_forest", "gradient_boosting", "hist_gradient_boosting",
                        "bayesian", "neural"]
                       if k in forecasts}
        viz.save(viz.forecast_comparison(train, test, headline_fc),
                 config.OUT_FIGURES / "forecast_comparison.png")
        viz.save(viz.error_diagnostics(test, headline_fc),
                 config.OUT_FIGURES / "error_diagnostics.png")
        viz.save(viz.residual_acf(sarimax_model.resid.iloc[config.SEASON + 1:]),
                 config.OUT_FIGURES / "residual_acf.png")
        if "sarimax" in intervals:
            lo, hi = intervals["sarimax"]
            viz.save(viz.prediction_intervals(test, sarimax_mean, lo, hi,
                                              title="SARIMAX 95% prediction interval"),
                     config.OUT_FIGURES / "prediction_intervals.png")
        log(f"Saved forecasts, metrics and figures under {config.ROOT / 'outputs'}")

    return {"forecasts": all_fc, "metrics": metrics, "intervals": intervals,
            "sarimax_model": sarimax_model, "feature_models": feat_fitted,
            "train": train, "test": test}
