"""Put src/ on the path for pytest, and provide a synthetic weekly series so the
tests need no network."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def weekly_series():
    idx = pd.date_range("2015-01-04", periods=300, freq="W")
    rng = np.random.default_rng(0)
    week = idx.isocalendar().week.astype(int).to_numpy()
    load = 55 + 6 * np.cos(2 * np.pi * (week - 3) / 52) + rng.normal(0, 1, len(idx))
    return pd.Series(load, index=idx, name="load_gw")
