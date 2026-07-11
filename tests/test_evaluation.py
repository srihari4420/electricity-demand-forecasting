import numpy as np
from electricity_demand import evaluation as ev


def test_mase_zero_for_perfect_forecast(weekly_series):
    train, test = weekly_series.iloc[:-104], weekly_series.iloc[-104:]
    assert ev.mase(test.values, test.values, train.values) == 0.0


def test_mae_rmse_zero_for_perfect_forecast(weekly_series):
    test = weekly_series.iloc[-104:]
    assert ev.mae(test.values, test.values) == 0.0
    assert ev.rmse(test.values, test.values) == 0.0


def test_bias_sign(weekly_series):
    test = weekly_series.iloc[-104:]
    assert ev.bias(test.values, test.values + 3) > 0    # forecasting high -> positive bias
    assert ev.bias(test.values, test.values - 3) < 0
