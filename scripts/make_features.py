"""Build the processed modelling table (target + calendar, holiday, temperature,
lag and rolling features) and save it under data/processed."""
import _bootstrap  # noqa: F401
import pandas as pd
from electricity_demand import data, features, config


def main():
    config.ensure_dirs()
    weekly_path = config.DATA_PROCESSED / "load_weekly_gw.parquet"
    temp_path = config.DATA_RAW / "temperature_berlin_daily.parquet"

    if weekly_path.exists():
        weekly = pd.read_parquet(weekly_path)[config.TARGET]
    else:
        weekly, _, _ = data.build_weekly_load(save=True)

    if temp_path.exists():
        temp = pd.read_parquet(temp_path)
    else:
        temp = data.download_temperature(start=str(weekly.index.min().date()),
                                         end=str(weekly.index.max().date()))

    table = features.build_feature_table(weekly, temp_daily=temp, add_holidays=True)
    table.to_parquet(config.DATA_PROCESSED / "features.parquet")
    table.to_csv(config.DATA_PROCESSED / "features.csv")
    print(f"Feature table: {table.shape} -> {config.DATA_PROCESSED / 'features.parquet'}")


if __name__ == "__main__":
    main()
