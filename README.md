# Retail Sales Forecasting: Store 1 GROCERY I

## Overview

This project forecasts daily sales for Store 1 and the GROCERY I product family from the Corporation Favorita retail sales dataset. The goal is to generate a fixed-origin 28-day forecast and compare the performance of simple reference baselines, a classical statistical time-series model (SARIMA), and a machine learning model (XGBoost).

## Problem Setup

- **Target Series**: Daily sales units (`sales`)
- **Store**: Store 1 (located in Quito, Ecuador)
- **Product Family**: GROCERY I
- **Forecast Origin**: 2017-07-18
- **Training Period**: 2013-01-01 to 2017-07-18 (1,660 calendar days)
- **Holdout Test Period**: 2017-07-19 to 2017-08-15 (28 calendar days)
- **Forecast Horizon**: Exactly 28 days

All models are evaluated strictly out-of-sample from the 2017-07-18 origin without updating or refitting on actual test-period observations.

## Data Preparation

The raw sales data was transformed into a modeling-ready, daily time series in `notebook/02_data_preparation.ipynb`:

- **Continuous Daily Calendar**: The series was reindexed to an unbroken daily calendar (`freq='D'`).
- **Missing Date Treatment**: Four calendar dates were absent from the raw series: December 25 in 2013, 2014, 2015, and 2016. Because these dates coincide with the national Christmas Day holiday, restoring them with zero sales represents a modeling assumption based on the project analysis rather than definitive proof of store closure. An indicator column `was_missing` flags these four restored dates.
- **Observed Zero Sales**: Six observed zero-sales dates were present in the raw data (five New Year's Days from 2013 to 2017, and July 7, 2015). These records are preserved as actual zero-sales observations and are distinct from the restored missing dates.
- **Engineered Features**: Features constructed for tabular modeling include:
  - Calendar features: `day_of_week`, `day_of_month`, `month`, `quarter`, `year`, `is_weekend`
  - Promotional feature: `onpromotion` (count of family items on promotion)
  - Missing flag: `was_missing`
  - Lag features: `lag_1`, `lag_7`, `lag_14`, `lag_28`
  - Rolling mean features: `rolling_mean_7`, `rolling_mean_28` (computed on shifted sales to prevent leakage)
- **Burn-In Handling**: Because `lag_28` and `rolling_mean_28` require 28 days of history, the first 28 days of the training set (2013-01-01 to 2013-01-28) contain `NaN` values and are dropped when fitting supervised tabular models, leaving 1,632 complete training observations.
- **Recursive Forecasting Design**: To prevent data leakage during multi-step test forecasting, short lag and rolling features were not populated with actual test sales. Instead, predictions were generated recursively, feeding each day's model prediction into subsequent lag and rolling calculations.

## Models

1. **Naive Baseline**: Emits a constant prediction equal to the sales on the forecast origin ($y_{\text{2017-07-18}} = 3,370.0$) across all 28 test dates.
2. **Seasonal Naive Baseline**: Emits a periodic forecast by repeating the day-of-week sales values from the final week of training data (2017-07-12 to 2017-07-18) across the four 7-day test cycles.
3. **SARIMA**: A classical Seasonal Autoregressive Integrated Moving Average model specified as $\text{SARIMA}(1, 1, 1) \times (1, 1, 1)_7$, capturing both non-seasonal dynamics and weekly seasonality ($s=7$) estimated via maximum likelihood on training data only.
4. **XGBoost**: A gradient-boosted decision tree regressor (`n_estimators=100`, `learning_rate=0.05`, `max_depth=4`, `random_state=42`) using recursive multi-step forecasting across the 28-day holdout horizon.

## Results

Evaluation metrics:
- **MAE**: Mean Absolute Error across all 28 test observations.
- **RMSE**: Root Mean Squared Error across all 28 test observations.
- **MAPE**: Mean Absolute Percentage Error, calculated strictly on observations where actual sales $\gt 0$.

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Naive | 1025.96 | 1212.59 | 62.21% |
| Seasonal Naive | 361.96 | 527.63 | 18.01% |
| SARIMA | 420.19 | 552.78 | 22.36% |
| XGBoost | 304.51 | 443.19 | 16.42% |

### Factual Interpretation

- XGBoost achieved the lowest MAE, RMSE, and MAPE among the four models on this specific 28-day holdout.
- Seasonal Naive substantially outperformed the plain Naive baseline, demonstrating that accounting for weekly seasonality is critical for this series.
- SARIMA did not outperform Seasonal Naive on this holdout.
- These results reflect performance on this specific forecast period and series and should not be generalized beyond them.

## Error Analysis

A detailed error analysis was conducted in `notebook/06_error_analysis.ipynb`:

- **Error Definition and Bias**: Forecast error is defined as $\text{error} = \text{actual} - \text{prediction}$. Under this convention, negative error indicates overprediction and positive error indicates underprediction. XGBoost had a mean error of -212.21 units across the holdout, indicating that its predictions were higher than actual sales on average.
- **Largest-Error Dates**: The largest XGBoost absolute error occurred on Friday, 2017-08-11 (actual sales: 1,270.0, prediction: 2,762.96, error: -1,492.96). This date was also the largest absolute error date for Seasonal Naive (-1,912.00) and SARIMA (-1,744.75). The second largest XGBoost error occurred on Saturday, 2017-08-12 (actual sales: 1,630.0, prediction: 2,682.76, error: -1,052.76).
- **Weekday Analysis**: In this 28-day holdout, XGBoost had relatively low MAE on Sundays (178.22) and Wednesdays (188.38), while Saturdays (525.23) and Fridays (448.74) had higher MAE. Because each weekday has only four observations in this 28-day holdout, this does not establish general weekday performance.
- **Promotion Analysis**: Every day in the 28-day holdout had at least 19 items on promotion in this family (zero non-promotion days). In an exploratory split at the median number of promoted items in this holdout (44.5 items), days with $\ge 44.5$ promoted items had an MAE of 194.05 and a mean error of -39.45, while days with $< 44.5$ items had an MAE of 414.96 and a mean error of -384.97. This analysis is observational; we cannot claim that promotions caused higher or lower forecast errors.
- **Earthquake Context**: During exploratory analysis, an external shock was noted following the April 16, 2016 earthquake in Ecuador, which caused temporary sales increases in mid-2016. Because that event occurred 15 months prior to the holdout period, it serves only as historical series context and is not an explanation for 2017 forecast errors.

## Limitations

- **Single Series Scope**: The project evaluates only one store (Store 1) and one product family (GROCERY I).
- **Single Holdout Window**: Performance is assessed on a single 28-day holdout period rather than across multiple rolling forecast origins or backtest splits.
- **No Hyperparameter Optimization**: Models were fit using straightforward, standard configurations without systematic hyperparameter search.
- **Limited External Variables**: The feature set is limited to calendar variables, lag/rolling terms, and item promotion counts. Exogenous factors such as item-level prices, competitor activity, or weather are not included.
- **Observational Promotion Analysis**: Promotional relationships are purely observational and do not support causal interpretations.
- **Modeling Assumption on Missing Dates**: Imputing missing Christmas dates with zero sales is a modeling assumption rather than confirmed store closure data.
- **Generalizability**: Findings are specific to this store, product family, and holdout window, and may not generalize to other stores, categories, or time periods.

## Project Structure

```
.
├── data/
│   ├── holidays_events.csv
│   ├── processed_store1_grocery1.csv
│   ├── stores.csv
│   └── train.csv
├── notebook/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_baseline_forecasting.ipynb
│   ├── 04_sarima_forecasting.ipynb
│   ├── 05_xgboost_forecasting.ipynb
│   └── 06_error_analysis.ipynb
├── outputs/
│   ├── baseline_results.csv
│   └── model_comparison.csv
├── requirements.txt
└── README.md
```

## Reproducibility

### Environment Setup

Create and activate a virtual environment, then install the project dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running the Notebooks

Execute the notebooks in sequential order:

1. `notebook/01_exploratory_data_analysis.ipynb`: Explores the raw series, missing values, seasonality, and promotional counts.
2. `notebook/02_data_preparation.ipynb`: Reindexes to a continuous daily calendar, handles missing dates, and constructs lag/rolling features.
3. `notebook/03_baseline_forecasting.ipynb`: Fits and evaluates the Naive and Seasonal Naive reference baselines on the 28-day holdout.
4. `notebook/04_sarima_forecasting.ipynb`: Fits SARIMA(1,1,1)x(1,1,1,7) on training data and generates a 28-day out-of-sample forecast.
5. `notebook/05_xgboost_forecasting.ipynb`: Fits an XGBoost regressor and produces a 28-day recursive forecast without test-period leakage.
6. `notebook/06_error_analysis.ipynb`: Compares all four models, analyzes residual patterns across time, weekdays, and promotional volume, and runs validation checks.

## Conclusion

Comparing these four approaches over the 28-day holdout illustrates the progression from simple reference baselines to classical time-series and machine learning models:
- Incorporating 7-day weekly seasonality through Seasonal Naive reduced MAE from 1025.96 to 361.96 compared to the constant Naive baseline.
- SARIMA captured the weekly cycle but did not improve upon Seasonal Naive on this specific holdout (MAE 420.19 vs. 361.96).
- Recursive XGBoost achieved the lowest error metrics on this holdout (MAE 304.51, MAPE 16.42%) by combining seasonal lags, calendar indicators, and promotional counts within a leakage-safe multi-step setup.
- All models exhibited negative mean errors over this 28-day window, indicating that predictions were higher than actual sales on average.
