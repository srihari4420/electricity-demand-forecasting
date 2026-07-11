"""Feature-based machine-learning models on the supervised table: a linear
baseline, Random Forest, and two flavours of gradient boosting."""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor)

from .. import config


def build_models(seed=config.RANDOM_SEED):
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=3, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.03, max_leaf_nodes=15, random_state=seed),
    }


def fit_predict(X_train, y_train, X_test, seed=config.RANDOM_SEED):
    """Fit each model and return its test-period forecast plus the fitted object.
    The best single model (by training-set fit is a poor guide, so we simply
    return them all and let the evaluation pick) is chosen downstream."""
    preds, fitted = {}, {}
    for name, model in build_models(seed).items():
        model.fit(X_train, y_train)
        preds[name] = pd.Series(model.predict(X_test), index=X_test.index, name=name)
        fitted[name] = model
    return preds, fitted


def feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    if hasattr(model, "coef_"):
        return pd.Series(model.coef_, index=feature_names).sort_values(key=abs, ascending=False)
    return None
