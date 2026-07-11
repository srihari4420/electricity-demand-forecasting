# Forecasting German Electricity Demand

This repository contains a reproducible time-series forecasting pipeline for modelling and forecasting German electricity demand.

The project uses hourly electricity load data from Open Power System Data, aggregates it to weekly average load, and compares a range of forecasting models, from simple statistical benchmarks to more complex machine-learning, Bayesian, and neural forecasting methods.

## Project aim

The aim of this project is to forecast weekly German electricity demand and compare the performance, interpretability, and complexity of different forecasting approaches.

The main research questions are:

1. How well do simple benchmark methods forecast weekly electricity demand?
2. Does a SARIMAX model improve on seasonal benchmarks?
3. Do temperature and holiday covariates improve forecast accuracy?
4. Do feature-based or neural models justify their additional complexity?
5. Which model would be most appropriate for an operational forecasting setting?

## Data

The target series is German electricity load from Open Power System Data.

The original data are hourly electricity load observations. These are cleaned, aggregated to weekly average load, and converted from MW to GW.

The main target variable is:

```text
load_gw
```

Optional covariates may include:

```text
temp_mean
temp_min
temp_max
heating_degree_days
cooling_degree_days
holiday_days
has_holiday
```

Temperature features are external covariates and should be treated carefully. In a real forecasting setting, future temperature would not be known exactly and would need to come from a weather forecast. If realised future temperature is used in the test period, the resulting forecast should be described as a conditional forecast.

Holiday features are generally known in advance and are valid future covariates.

## Repository structure

```text
electricity-demand-forecasting/
│
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── src/
│   └── electricity_demand/
│       ├── __init__.py
│       ├── config.py
│       ├── pipeline.py
│       ├── data.py
│       ├── features.py
│       ├── evaluation.py
│       ├── plotting.py
│       └── models/
│           ├── __init__.py
│           ├── benchmarks.py
│           ├── sarimax.py
│           ├── feature_models.py
│           ├── bayesian.py
│           └── neural.py
│
├── scripts/
│   ├── download_data.py
│   ├── make_features.py
│   ├── run_pipeline.py
│   └── evaluate_models.py
│
├── outputs/
│   ├── figures/
│   ├── forecasts/
│   ├── metrics/
│   └── model_objects/
│
├── reports/
│   ├── report.md
│   └── figures/
│
└── tests/
    ├── test_features.py
    ├── test_evaluation.py
    └── test_benchmarks.py
```

## Pipeline overview

The main modelling workflow is controlled by:

```text
scripts/run_pipeline.py
```

This script calls the reusable pipeline function in:

```text
src/electricity_demand/pipeline.py
```

The pipeline should perform the following steps:

1. Download or load the electricity demand data.
2. Clean the hourly load series.
3. Aggregate hourly load to weekly average demand.
4. Create calendar, holiday, temperature, lag, and rolling-window features.
5. Split the data into training and test periods.
6. Fit benchmark forecasting models.
7. Fit SARIMAX models.
8. Fit feature-based machine-learning models.
9. Optionally fit Bayesian and neural forecasting models.
10. Generate forecasts for the test period.
11. Evaluate all forecasts using common metrics.
12. Save forecasts, metrics, and figures.

The preferred command-line workflow is:

```bash
python scripts/run_pipeline.py
```

## Installation

Create and activate a virtual environment.

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Alternatively, using conda:

```bash
conda env create -f environment.yml
conda activate electricity-demand-forecasting
```

## Reproducing the analysis

From a fresh clone of the repository, the full workflow should run using:

```bash
python scripts/run_pipeline.py
```

The pipeline should create or update the following outputs:

```text
outputs/forecasts/all_forecasts.csv
outputs/metrics/model_comparison.csv
outputs/figures/forecast_comparison.png
outputs/figures/error_diagnostics.png
```

If the raw data are not stored in the repository, they should be recreated using:

```bash
python scripts/download_data.py
```

Then the processed modelling dataset can be created using:

```bash
python scripts/make_features.py
```

## Models

The project should compare at least the following forecasting methods.

### Benchmark models

The benchmark models should include:

```text
Mean forecast
Naive forecast
Seasonal naive forecast
Drift forecast
```

The seasonal naive model is especially important because weekly electricity demand has strong annual seasonality.

### SARIMAX model

The SARIMAX model should include weekly seasonality. A reasonable starting point is:

```python
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 52)
```

Students should justify the choice of differencing orders and seasonal period. If exogenous variables are used, they should be passed through the `exog` argument.

Example covariates include:

```text
temperature features
heating degree days
cooling degree days
holiday indicators (optional)
```

### Feature-based machine-learning model

The feature-based model should use a supervised learning table containing lag, rolling-window, calendar, holiday, and temperature features.

Possible models include:

```text
linear regression
random forest
gradient boosting
histogram gradient boosting
XGBoost or LightGBM, if installed
```

Students must ensure that lag and rolling features use only past information.

### Bayesian model

The Bayesian model may be a regression model with seasonal, holiday, and temperature covariates.

Students should discuss:

```text
posterior uncertainty
posterior predictive intervals
prior assumptions
interpretability of coefficients
```

### Neural or foundation-model-style model

Students may include a neural forecasting model such as:

```text
N-BEATS
N-HiTS
LSTM
Temporal Fusion Transformer
```

Students should justify whether the amount of data is sufficient for the chosen model. For weekly data, neural models may overfit and may not outperform simpler approaches.

## Evaluation

All models should be evaluated on the same test period.

Required metrics:

```text
MAE
RMSE
MASE
Bias
```

Recommended additional diagnostics:

```text
forecast error plots
residual autocorrelation
prediction interval coverage
performance around Christmas and New Year
performance during unusually hot or cold weeks
```

Students should compare all models against the seasonal naive benchmark.

## Train-test split

The default test set should be the final 104 weeks of the series.

This corresponds to a two-year forecasting evaluation period.

Students should not use a random train-test split, because this is a time-series forecasting problem.

A stronger analysis may also include rolling-origin evaluation.

## Data leakage

Students must be careful to avoid data leakage.

Examples of leakage include:

```text
using future target values in lag or rolling features
scaling the full dataset before the train-test split
using observed future temperature without describing the forecast as conditional
selecting models based directly on test-set performance
```

All preprocessing steps that learn from the data should be fitted on the training set only.

## Outputs

The pipeline should save forecasts to:

```text
outputs/forecasts/all_forecasts.csv
```

This file should contain one row per test-period timestamp and columns such as:

```text
actual
mean
naive
seasonal_naive
drift
sarimax
feature_model
bayesian
neural
```

The pipeline should save model comparison metrics to:

```text
outputs/metrics/model_comparison.csv
```

This file should contain columns such as:

```text
model
MAE
RMSE
MASE
Bias
```

The pipeline should save figures to:

```text
outputs/figures/
```

Recommended figures include:

```text
forecast_comparison.png
error_diagnostics.png
residual_acf.png
prediction_intervals.png
```

## Report

The final report should summarise the analysis and discuss the modelling choices.

A suggested report structure is:

```text
1. Introduction
2. Data and preprocessing
3. Exploratory analysis
4. Forecasting methods
5. Evaluation design
6. Results
7. Error analysis
8. Discussion
9. Limitations
10. Conclusion
```

The report should answer the following questions:

1. Which model performs best?
2. Does any complex model meaningfully improve on the seasonal naive benchmark?
3. Do temperature and holiday features improve performance?
4. Are the more complex models justified?
5. Which model would be recommended for operational use?
6. What are the main limitations of the analysis?

## Testing

The repository should include simple tests for key functions.

Examples:

```text
test that forecast lengths match the test period
test that MASE is zero for a perfect forecast
test that lag features do not use future values
test that the processed dataset has no missing target values
```

Run tests using:

```bash
pytest
```

## Good practice

Students should follow these principles:

```text
Use clear function names.
Keep reusable code in src/.
Keep notebooks for exploration and explanation.
Do not commit large raw data files.
Make the pipeline reproducible from a fresh clone.
Set random seeds where relevant.
Compare every advanced model against simple benchmarks.
Explain whether covariates are known at the forecast origin.
```

## Expected submission

The submitted repository should include:

```text
README.md
requirements.txt or environment.yml
source code in src/
pipeline script in scripts/
notebooks showing exploration and results
generated metrics and figures
final report
```

The repository should run from a fresh clone using the instructions in this README.
