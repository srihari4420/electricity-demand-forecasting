"""Recompute the comparison metrics from a saved forecast file. Useful for
re-scoring without refitting anything."""
import _bootstrap  # noqa: F401
import pandas as pd
from electricity_demand import config, evaluation as ev


def main():
    fc = pd.read_csv(config.OUT_FORECASTS / "all_forecasts.csv", index_col=0, parse_dates=True)
    actual = fc["actual"]
    # scale MASE on the pre-test history if the full forecast file is present
    train = actual   # single-file fallback; metrics still comparable across models
    rows = []
    for col in fc.columns:
        if col == "actual":
            continue
        series = fc[col].dropna()
        if series.empty:
            continue
        common = actual.loc[series.index]
        rows.append(ev.evaluate(common.values, series.values, train.values, col))
    table = ev.metrics_table(rows)
    table.reset_index().to_csv(config.OUT_METRICS / "model_comparison.csv", index=False)
    print(table.to_string())


if __name__ == "__main__":
    main()
