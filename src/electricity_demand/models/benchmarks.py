"""The four reference forecasts. Seasonal naive is the one that matters, because
weekly demand has such a strong annual cycle."""

import numpy as np
import pandas as pd

from .. import config


class BenchmarkForecaster:
    def __init__(self, train, season=config.SEASON):
        self.train = train
        self.season = season

    def mean(self, index):
        return pd.Series(self.train.mean(), index=index, name="mean")

    def naive(self, index):
        return pd.Series(self.train.iloc[-1], index=index, name="naive")

    def seasonal_naive(self, index):
        last = self.train.iloc[-self.season:].values
        reps = int(np.ceil(len(index) / self.season))
        return pd.Series(np.tile(last, reps)[:len(index)], index=index, name="seasonal_naive")

    def drift(self, index):
        n = len(self.train)
        slope = (self.train.iloc[-1] - self.train.iloc[0]) / (n - 1)
        steps = np.arange(1, len(index) + 1)
        return pd.Series(self.train.iloc[-1] + slope * steps, index=index, name="drift")

    def all(self, index):
        return {"mean": self.mean(index), "naive": self.naive(index),
                "seasonal_naive": self.seasonal_naive(index), "drift": self.drift(index)}
