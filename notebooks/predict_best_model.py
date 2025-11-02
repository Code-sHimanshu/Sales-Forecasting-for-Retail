import os
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# === PATHS ===
base_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(base_dir, "data", "processed")
models_dir = os.path.join(base_dir, "models")
predictions_dir = os.path.join(base_dir, "data", "predictions")

os.makedirs(predictions_dir, exist_ok=True)

print("Loading trained models and scaler...")
lr_model = joblib.load(os.path.join(models_dir, "Linear_Regression.joblib"))
rf_model = joblib.load(os.path.join(models_dir, "Random_Forest.joblib"))
xgb_model = joblib.load(os.path.join(models_dir, "XGBoost.joblib"))
scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))

# === LOAD TEST DATA ===
print("Loading test data...")
test_path = os.path.join(data_dir, "test_preprocessed.csv")
test_df = pd.read_csv(test_path)

drop_cols = [col for col in ['Date', 'IsHoliday_x', 'IsHoliday_y', 'IsHoliday'] if col in test_df.columns]
if drop_cols:
    print(f"Dropping non-numeric columns: {drop_cols}")
    test_df.drop(columns=drop_cols, inplace=True)

# === HANDLE MISSING VALUES ===
missing_count = test_df.isnull().sum().sum()
if missing_count > 0:
    print(f"Found {missing_count} missing values. Imputing with column means...")
    test_df = test_df.fillna(test_df.mean(numeric_only=True))
else:
    print("No missing values found.")

# === ALIGN FEATURES WITH TRAINING SCALER ===
print("Aligning test features with training columns...")
scaler_features = scaler.feature_names_in_

# Add missing columns (fill with 0)
missing_features = [col for col in scaler_features if col not in test_df.columns]
for col in missing_features:
    test_df[col] = 0

# Drop extra columns not seen during training
extra_features = [col for col in test_df.columns if col not in scaler_features]
if extra_features:
    print(f"Removing extra columns not seen during training: {extra_features}")
    test_df = test_df.drop(columns=extra_features)

# Reorder columns to match training
test_df = test_df[scaler_features]

# === SCALING ===
print("Scaling test data...")
X_test_scaled = scaler.transform(test_df)

# === PREDICT USING ALL MODELS ===
print("Generating predictions...")
predictions = {}

try:
    predictions['Linear_Regression'] = lr_model.predict(X_test_scaled)
except Exception as e:
    print(f"⚠️ Linear Regression failed: {e}")

try:
    predictions['Random_Forest'] = rf_model.predict(X_test_scaled)
except Exception as e:
    print(f"⚠️ Random Forest failed: {e}")

try:
    predictions['XGBoost'] = xgb_model.predict(X_test_scaled)
except Exception as e:
    print(f"⚠️ XGBoost failed: {e}")

# === SAVE ALL PREDICTIONS ===
for model_name, preds in predictions.items():
    output_path = os.path.join(predictions_dir, f"{model_name}_predictions.csv")
    pd.DataFrame(preds, columns=["Weekly_Sales"]).to_csv(output_path, index=False)
    print(f"✅ Saved {model_name} predictions to: {output_path}")

print("\nAll predictions generated successfully.")
