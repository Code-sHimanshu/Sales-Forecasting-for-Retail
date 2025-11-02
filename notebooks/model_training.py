"""
model_training.py

Trains multiple regression models for weekly sales forecasting,
evaluates their performance, and saves models, metrics, and plots.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ====================================
# 1. PATH CONFIGURATION
# ====================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
PLOTS_DIR = os.path.join(REPORTS_DIR, 'plots')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ====================================
# 2. LOAD DATA
# ====================================
print("Loading processed training and test feature data...")

train_path = os.path.join(DATA_DIR, 'train_features.csv')
test_path = os.path.join(DATA_DIR, 'test_features.csv')

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# ====================================
# 3. DATA SPLIT
# ====================================
if 'Weekly_Sales' not in train_df.columns:
    raise KeyError("Target column 'Weekly_Sales' is missing in training data.")

X = train_df.drop(columns=['Weekly_Sales'])
y = train_df['Weekly_Sales']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Columns in training data:", X_train.columns)
print("Data types:\n", X_train.dtypes)

# ====================================
# 4. HANDLE NON-NUMERIC + MISSING VALUES + SCALING
# ====================================

# Drop non-numeric columns
non_numeric_cols = X_train.select_dtypes(exclude=['number']).columns.tolist()
if non_numeric_cols:
    print(f"\nDropping non-numeric columns before scaling: {non_numeric_cols}")
    X_train = X_train.drop(columns=non_numeric_cols)
    X_val = X_val.drop(columns=non_numeric_cols)

# Handle missing values
missing_cols = X_train.columns[X_train.isna().any()].tolist()
if missing_cols:
    print(f"Imputing missing values in columns: {missing_cols}")
    for col in missing_cols:
        median_val = X_train[col].median()
        X_train[col].fillna(median_val, inplace=True)
        X_val[col].fillna(median_val, inplace=True)

# Scale numeric columns
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Save scaler for inference
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))


# ====================================
# 5. MODEL DEFINITIONS
# ====================================
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
}

# ====================================
# 6. TRAINING AND EVALUATION
# ====================================
metrics = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    if name == "Linear Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    metrics[name] = {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4)
    }

    # Save model
    model_filename = os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.joblib")
    joblib.dump(model, model_filename)

    # ======================
    # Feature Importance
    # ======================
    if hasattr(model, "feature_importances_"): # Expected indented block
        importance = model.feature_importances_
        # Use the actual feature columns used in training
        used_features = X_train.columns
        importance_df = pd.DataFrame({
            'Feature': used_features,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)


        plt.figure(figsize=(10, 5))
        plt.barh(importance_df['Feature'][:15][::-1],
            importance_df['Importance'][:15][::-1])
        plt.title(f"Top 15 Features - {name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{name.replace(' ', '_')}_feature_importance.png"))
        plt.close()


    # ======================
    # Predictions vs Actuals
    # ======================
    plt.figure(figsize=(6, 6))
    plt.scatter(y_val, y_pred, alpha=0.4, color='blue')
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title(f"Actual vs Predicted Sales - {name}")
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{name.replace(' ', '_')}_actual_vs_predicted.png"))
    plt.close()

# ====================================
# 7. SAVE METRICS
# ====================================
with open(os.path.join(REPORTS_DIR, 'model_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=4)

print("\n✅ Training completed successfully!")
print(json.dumps(metrics, indent=4))
print(f"\nAll outputs saved in:\n- Models: {MODELS_DIR}\n- Reports: {REPORTS_DIR}")
