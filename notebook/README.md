# Operating Profit Prediction Using ANN

## Project Overview

This project uses an Artificial Neural Network (ANN) to predict operating profit before tax using enterprise survey data.

## Dataset

The dataset contains enterprise information from 2011 to 2025, including:

- Year
- Industry
- Enterprise size
- Industry code
- Operating profit before tax
- Other enterprise-related information

## Data Preparation

The following steps were performed:

- Data cleaninggit init
- Removal of unnecessary columns
- Handling of non-numeric values
- Target variable selection
- Categorical encoding
- Duplicate checking
- Train-test splitting
- Feature scaling

## Model

An Artificial Neural Network was built using TensorFlow/Keras.

Architecture:

- Input layer: 48 features
- Dense layer: 64 neurons, ReLU
- Dense layer: 32 neurons, ReLU
- Dense layer: 16 neurons, ReLU
- Output layer: 1 neuron

Optimizer: Adam

Loss function: Mean Squared Error

## Results

Test MAE: 479.02

Test RMSE: 1571.48

R² Score: 0.9541

The model achieved an R² score of approximately 95.4% on the test dataset.

## Visualizations

The project includes:

- Training and validation loss
- Actual vs predicted operating profit

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow
- Keras

## Project Structure

```text
ANN_PROJECT/
├── data/
├── notebook/
├── operating_profit_ann.keras
├── scaler.pkl
├── cleaned_operating_profit_data.csv
├── actual_vs_predicted.png
├── training_loss.png
└── README.md