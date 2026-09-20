# ANN Operating Profit Prediction

An Artificial Neural Network (ANN) project that predicts operating profit before tax using enterprise survey data.

## Project Overview

This project uses Python, Pandas, Scikit-learn, TensorFlow, and Keras to prepare enterprise survey data and train an ANN regression model.

The dataset contains enterprise information from 2011 to 2025, including:

- Year
- Industry
- Enterprise size
- Industry code
- Operating profit before tax

## Data Preparation

The project includes:

- Data cleaning
- Removal of unnecessary columns
- Target variable selection
- Categorical encoding
- Duplicate checking
- Train/test splitting
- Feature scaling with StandardScaler

## ANN Model

The model was built using TensorFlow/Keras.

Architecture:

- Input: 48 features
- Dense layer: 64 neurons, ReLU
- Dense layer: 32 neurons, ReLU
- Dense layer: 16 neurons, ReLU
- Output: 1 neuron

The model uses the Adam optimizer and Mean Squared Error (MSE) loss.

## Results

Test MAE: **479.02**

Test RMSE: **1571.48**

R² Score: **0.9541**

The model achieved an R² score of approximately **95.4%** on the test dataset.

## Visualizations

The project includes:

- Training and validation loss
- Actual vs. predicted operating profit

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow
- Keras

## Project Structure

```
ANN_PROJECT/
├── data/
│   └── annual-enterprise-survey-2025-financial-year-provisional.csv
├── notebook/
│   ├── 01_data_understanding.ipynb
│   ├── cleaned_operating_profit_data.csv
│   ├── operating_profit_ann.keras
│   ├── scaler.pkl
│   ├── actual_vs_predicted.png
│   └── training_loss.png
└── README.md
```

## Purpose

This project demonstrates an end-to-end machine learning workflow, from data understanding and preprocessing to ANN training, evaluation, and visualization.
