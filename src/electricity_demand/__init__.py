"""Forecasting German electricity demand: a modular time-series pipeline."""

from . import config, data, features, evaluation, plotting, pipeline
from .pipeline import run_pipeline

__version__ = "0.1.0"
__all__ = ["config", "data", "features", "evaluation", "plotting", "pipeline",
           "run_pipeline"]
