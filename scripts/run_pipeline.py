"""Run the full forecasting comparison end to end and save forecasts, metrics and
figures. This is the main command-line entry point:

    python scripts/run_pipeline.py
"""
import argparse
import _bootstrap  # noqa: F401
from electricity_demand import run_pipeline


def main():
    ap = argparse.ArgumentParser(description="Forecast German electricity demand.")
    ap.add_argument("--no-bayesian", action="store_true", help="skip the Bayesian model")
    ap.add_argument("--no-neural", action="store_true", help="skip the neural model")
    ap.add_argument("--no-save", action="store_true", help="do not write outputs to disk")
    args = ap.parse_args()

    result = run_pipeline(
        with_bayesian=not args.no_bayesian,
        with_neural=False if args.no_neural else None,
        save=not args.no_save,
    )
    print("\n=== Model comparison (sorted by RMSE) ===")
    print(result["metrics"].to_string())


if __name__ == "__main__":
    main()
