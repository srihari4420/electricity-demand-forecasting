import numpy as np
from electricity_demand import features as feat, config


def test_lag_features_do_not_use_future(weekly_series):
    lags = feat.lag_features(weekly_series, lags=(1,), roll_windows=())
    # lag_1 at time t must equal the target at t-1
    aligned = lags["lag_1"].dropna()
    assert np.allclose(aligned.values, weekly_series.shift(1).dropna().values)


def test_rolling_mean_excludes_current_week(weekly_series):
    lags = feat.lag_features(weekly_series, lags=(), roll_windows=(4,))
    # a shift(1) rolling(4) mean at t uses weeks t-4..t-1, never t
    manual = weekly_series.shift(1).rolling(4).mean()
    assert np.allclose(lags["roll_mean_4"].dropna().values, manual.dropna().values)


def test_processed_table_has_no_missing_target(weekly_series):
    table = feat.build_feature_table(weekly_series, temp_daily=None, add_holidays=True)
    assert table[config.TARGET].notna().all()
    assert not table.isna().any().any()
