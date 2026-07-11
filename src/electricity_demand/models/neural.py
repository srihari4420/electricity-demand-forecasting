"""Optional neural model: a small LSTM over weekly lag-windows. Weekly data is
short (a couple of hundred training points), so this model is expected to
struggle to beat the simpler approaches and is prone to overfitting, which is
itself a finding worth reporting. TensorFlow is imported lazily so the rest of
the pipeline runs without it."""

import numpy as np
import pandas as pd

from .. import config

try:
    import tensorflow as tf
    _HAS_TF = True
except ImportError:
    _HAS_TF = False


def available():
    return _HAS_TF


class WeeklyLSTM:
    def __init__(self, lookback=52, units=32, dropout=0.1, epochs=200,
                 seed=config.RANDOM_SEED):
        if not _HAS_TF:
            raise ImportError("TensorFlow is not installed; skip the neural model.")
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.seed = seed

    def _scale_fit(self, y):
        self.y_min, self.y_max = float(y.min()), float(y.max())
        return (y - self.y_min) / (self.y_max - self.y_min)

    def _scale_apply(self, y):
        return (y - self.y_min) / (self.y_max - self.y_min)

    def _unscale(self, y):
        return y * (self.y_max - self.y_min) + self.y_min

    def _windows(self, arr):
        X, y = [], []
        for i in range(len(arr) - self.lookback):
            X.append(arr[i:i + self.lookback])
            y.append(arr[i + self.lookback])
        return np.array(X).reshape(-1, self.lookback, 1), np.array(y)

    def fit(self, train_series):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping

        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        self.train_series = train_series
        scaled = self._scale_fit(train_series).values          # scaling fitted on train only
        X, y = self._windows(scaled)

        model = Sequential([
            LSTM(self.units, input_shape=(self.lookback, 1)),
            Dropout(self.dropout),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
        model.fit(X, y, validation_split=0.15, epochs=self.epochs, batch_size=16,
                  callbacks=[stop], verbose=0)
        self.model = model
        return self

    def forecast(self, test_index, full_series):
        """Recursive multi-step forecast across the test period, seeded from the
        training tail. The model is fed its own predictions, so error compounds,
        which is the honest way to score a long horizon."""
        scaled_full = self._scale_apply(full_series).values
        n_train = len(full_series) - len(test_index)
        window = scaled_full[n_train - self.lookback:n_train].copy()
        preds = []
        for _ in range(len(test_index)):
            yhat = self.model.predict(window.reshape(1, self.lookback, 1), verbose=0)[0, 0]
            preds.append(yhat)
            window = np.append(window[1:], yhat)
        return pd.Series(self._unscale(np.array(preds)), index=test_index, name="neural")
