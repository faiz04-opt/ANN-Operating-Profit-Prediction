import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

DATA_PATH = "data/annual-enterprise-survey-2025-financial-year-provisional-size-bands.csv"
OUT_DIR = "notebook"

np.random.seed(42)
tf.random.set_seed(42)

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")], errors="ignore")

profit_df = df[df["variable"].eq("Operating profit before tax")].copy()
profit_df["value"] = pd.to_numeric(profit_df["value"], errors="coerce")
profit_df = profit_df.dropna(subset=["value"])
profit_df = profit_df[~profit_df["rme_size_grp"].isin(["i_Industry_Total", "j_Grand_Total"])]
profit_df = profit_df.drop_duplicates()

train_df = profit_df[profit_df["year"] <= 2023]
val_df = profit_df[profit_df["year"] == 2024]
test_df = profit_df[profit_df["year"] == 2025]

features = ["year", "industry_name_ANZSIC", "rme_size_grp"]
target = "value"

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), ["year"]),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
     ["industry_name_ANZSIC", "rme_size_grp"]),
])

X_train = preprocessor.fit_transform(train_df[features])
X_val = preprocessor.transform(val_df[features])
X_test = preprocessor.transform(test_df[features])

target_scaler = StandardScaler()
y_train = target_scaler.fit_transform(train_df[[target]])
y_val = target_scaler.transform(val_df[[target]])

model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=30, restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1,
)

pred_scaled = model.predict(X_test, verbose=0)
pred = target_scaler.inverse_transform(pred_scaled).ravel()
actual = test_df[target].to_numpy(dtype=float)

mae = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))
r2 = r2_score(actual, pred)

print(f"2025 Test MAE: {mae:,.2f}")
print(f"2025 Test RMSE: {rmse:,.2f}")
print(f"2025 Test R²: {r2:.4f}")

model.save(f"{OUT_DIR}/operating_profit_ann.keras")
joblib.dump(preprocessor, f"{OUT_DIR}/preprocessor.pkl")
joblib.dump(target_scaler, f"{OUT_DIR}/target_scaler.pkl")
profit_df.to_csv(f"{OUT_DIR}/cleaned_operating_profit_data.csv", index=False)
