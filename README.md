# ANN Operating Profit Prediction

An Artificial Neural Network (ANN) regression project that predicts **Operating profit before tax** from annual enterprise survey data.

## Project Overview

The dataset contains aggregate enterprise observations from **2011–2025**, including year, industry, enterprise size band, and operating profit before tax.

The workflow covers data cleaning, confidential-value handling, removal of aggregate totals, categorical encoding, scaling, chronological splitting, ANN training, evaluation, visualization, and model export.

## Data Cleaning

The source contains `C` values for confidential/suppressed observations. These are converted to missing values and excluded from the regression target.

Published aggregate groups such as `i_Industry_Total` and `j_Grand_Total` are excluded so the model learns from enterprise-size bands rather than mixing size-band observations with aggregate totals.

## Evaluation Methodology

A random train/test split was avoided because the data is organized by year, industry, and size group. A random split could place closely related observations from the same periods in both training and test sets.

The corrected notebook uses:

- **Training:** 2011–2023
- **Validation:** 2024
- **Test:** 2025

The final test metrics therefore measure performance on a completely held-out year.

## Features

The model uses `year`, `industry_name_ANZSIC`, and `rme_size_grp`. The industry code is omitted because it duplicates the industry category information.

Categorical variables are one-hot encoded. Features are standardized using training data only. The target is also standardized using training data only and converted back to original units for final evaluation.

## ANN Architecture

- Input layer
- Dense: 64 neurons, ReLU
- Dense: 32 neurons, ReLU
- Dense: 16 neurons, ReLU
- Output: 1 neuron

Training uses Adam, MSE loss, validation data from 2024, and early stopping.

## Metrics

The notebook reports MAE, RMSE, and R².

R² is **not classification accuracy**. It should be reported as an R² value such as `0.95`, not as '95% accuracy'.

The previous random-split metrics are no longer presented as the final results. Run the corrected notebook to generate the current 2025 test metrics.

## Project Structure

```
ANN-Operating-Profit-Prediction/
├── data/
│   └── annual-enterprise-survey-2025-financial-year-provisional-size-bands.csv
├── notebook/
│   ├── 01_data_understanding.ipynb
│   ├── cleaned_operating_profit_data.csv
│   ├── operating_profit_ann.keras
│   ├── preprocessor.pkl
│   ├── target_scaler.pkl
│   ├── actual_vs_predicted.png
│   └── training_loss.png
└── README.md
```

## Technologies

Python, Pandas, NumPy, Matplotlib, Scikit-learn, TensorFlow, Keras, and Joblib.

## Purpose

This project demonstrates an end-to-end ANN regression workflow with a time-aware evaluation strategy for a more realistic estimate of future-year performance.