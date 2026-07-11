#download the berlin electricity load and temperature data and save it to the raw data folder
import _bootstrap  # noqa: F401
from electricity_demand import data, config


def main():
    config.ensure_dirs()
    print("Downloading electricity load ...")
    weekly, hourly, n_gaps = data.build_weekly_load(save=True)
    print(f"  hourly rows: {len(hourly)} (gaps filled: {n_gaps}) | weekly rows: {len(weekly)}")

    print("Downloading Berlin temperature ...")
    temp = data.download_temperature(start=str(weekly.index.min().date()),
                                     end=str(weekly.index.max().date()))
    temp.to_parquet(config.DATA_RAW / "temperature_berlin_daily.parquet")
    print(f"  temperature rows: {len(temp)}")
    print(f"Saved raw data under {config.DATA_RAW}")


if __name__ == "__main__":
    main()
