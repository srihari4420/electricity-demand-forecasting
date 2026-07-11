"""A light Bayesian regression on the feature table. BayesianRidge keeps the
project dependency-free while still giving what the brief asks for: posterior
uncertainty, predictive intervals, and interpretable coefficients. It places a
prior over the weights and returns a predictive standard deviation per point."""

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

from .. import config


class BayesianForecaster:
    def __init__(self):
        self.scaler = StandardScaler()      # fitted on train only
        self.model = BayesianRidge()

    def fit(self, X_train, y_train):
        Xs = self.scaler.fit_transform(X_train)
        self.model.fit(Xs, y_train)
        self.feature_names = list(X_train.columns)
        return self

    def predict(self, X_test, alpha=0.05):
        """Return the posterior mean and a 95% predictive interval."""
        Xs = self.scaler.transform(X_test)
        mean, std = self.model.predict(Xs, return_std=True)
        z = 1.959963985                     # 95% normal quantile
        idx = X_test.index
        mean = pd.Series(mean, index=idx, name="bayesian")
        lower = pd.Series(mean.values - z * std, index=idx)
        upper = pd.Series(mean.values + z * std, index=idx)
        return mean, lower, upper

    def coefficients(self):
        """Posterior mean of each coefficient, on the standardised scale, most
        influential first."""
        return pd.Series(self.model.coef_, index=self.feature_names).sort_values(
            key=abs, ascending=False)
