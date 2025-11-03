"""
model_evaluation_plots.py
Generates and saves visual evaluation reports for the Sales Forecasting project.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix

# =========================
# 🔹 Paths Setup
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "test_preprocessed.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "plots")
os.makedirs(REPORTS_DIR, exist_ok=True)

# =========================
# 🔹 Load Data & Model
# =========================
print("🔹 Loading data and model...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
df = pd.read_csv(DATA_PATH)

# Drop non-numeric/unwanted columns
non_numeric_cols = ["Date", "IsHoliday", "IsHoliday_x", "IsHoliday_y"]
df = df.drop(columns=[col for col in non_numeric_cols if col in df.columns], errors='ignore')

# Split features and target
target_col = "Weekly_Sales"
X = df.drop(columns=[target_col])
y = df[target_col]

# Align columns with scaler
if hasattr(scaler, 'feature_names_in_'):
    X = X[scaler.feature_names_in_]

# Scale
X_scaled = scaler.transform(X)

# Predict
y_pred = model.predict(X_scaled)

# =========================
# 🔹 Metrics
# =========================
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

metrics = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R²"],
    "Value": [mae, rmse, r2]
})
metrics.to_csv(os.path.join(REPORTS_DIR, "evaluation_metrics.csv"), index=False)
print("✅ Metrics saved!")

# =========================
# 🔹 Plots
# =========================

# 1️⃣ Actual vs Predicted
plt.figure(figsize=(8,6))
sns.scatterplot(x=y, y=y_pred, alpha=0.5)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.savefig(os.path.join(REPORTS_DIR, "actual_vs_predicted.png"))
plt.close()

# 2️⃣ Error Distribution
errors = y - y_pred
plt.figure(figsize=(8,6))
sns.histplot(errors, bins=50, kde=True)
plt.title("Error Distribution (Residuals)")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.savefig(os.path.join(REPORTS_DIR, "error_distribution.png"))
plt.close()

# 3️⃣ Feature Importance (for tree models)
if hasattr(model, "feature_importances_"):
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    plt.figure(figsize=(10,6))
    sns.barplot(data=importance.head(15), x="Importance", y="Feature")
    plt.title("Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "feature_importance.png"))
    plt.close()

# 4️⃣ Confusion Matrix (for regression discretized bins)
bins = np.linspace(y.min(), y.max(), 6)
y_true_bin = np.digitize(y, bins)
y_pred_bin = np.digitize(y_pred, bins)
cm = confusion_matrix(y_true_bin, y_pred_bin)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Binned Sales)")
plt.xlabel("Predicted Bin")
plt.ylabel("Actual Bin")
plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"))
plt.close()

print("✅ All evaluation plots saved successfully in reports/plots/")
