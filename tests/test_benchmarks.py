from electricity_demand.models import BenchmarkForecaster


def test_forecast_lengths_match_test_period(weekly_series):
    train, test = weekly_series.iloc[:-104], weekly_series.iloc[-104:]
    fc = BenchmarkForecaster(train).all(test.index)
    for name, series in fc.items():
        assert len(series) == len(test), name
        assert series.index.equals(test.index), name


def test_seasonal_naive_repeats_last_year(weekly_series):
    train, test = weekly_series.iloc[:-104], weekly_series.iloc[-104:]
    sn = BenchmarkForecaster(train).seasonal_naive(test.index)
    # first 52 forecast weeks are exactly the last 52 training weeks
    assert (sn.iloc[:52].values == train.iloc[-52:].values).all()


def test_mean_forecast_is_constant(weekly_series):
    train, test = weekly_series.iloc[:-104], weekly_series.iloc[-104:]
    m = BenchmarkForecaster(train).mean(test.index)
    assert m.nunique() == 1
