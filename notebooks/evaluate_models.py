import os
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
from sklearn.preprocessing import KBinsDiscretizer
import itertools

# ===========================
# 1️⃣ Paths setup
# ===========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")
TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "test_preprocessed.csv")
ALL_MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORT_PATH = os.path.join(BASE_DIR, "reports", "final_evaluation_report.json")
PLOTS_DIR = os.path.join(BASE_DIR, "reports", "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

# ===========================
# 2️⃣ Load Data and Model
# ===========================
print("🔹 Loading test data and model...")
test_df = pd.read_csv(TEST_DATA_PATH)
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

if "Weekly_Sales" not in test_df.columns:
    raise ValueError("❌ Target column 'Weekly_Sales' missing in test data!")

X_test = test_df.drop(columns=["Weekly_Sales"], errors='ignore')
y_test = test_df["Weekly_Sales"]

train_columns = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else X_test.columns
common_cols = [col for col in X_test.columns if col in train_columns]
X_test = X_test[common_cols]
X_test_scaled = scaler.transform(X_test)

# ===========================
# 3️⃣ Predict and Evaluate
# ===========================
print("🔹 Making predictions...")
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

metrics = {
    "Mean Absolute Error": mae,
    "Root Mean Squared Error": rmse,
    "R² Score": r2
}
with open(REPORT_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Metrics saved to:", REPORT_PATH)

# ===========================
# 4️⃣ Visualization
# ===========================
print("🔹 Generating plots...")

# --- Actual vs Predicted ---
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.xlabel("Actual Weekly Sales")
plt.ylabel("Predicted Weekly Sales")
plt.title("Actual vs Predicted Weekly Sales")
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, "actual_vs_predicted.png"))
plt.close()

# --- Residuals Distribution ---
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=30, kde=True)
plt.title("Residuals Distribution")
plt.xlabel("Residuals (Actual - Predicted)")
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, "residuals_distribution.png"))
plt.close()

# --- Feature Importance ---
if hasattr(model, "feature_importances_"):
    importance = pd.Series(model.feature_importances_, index=common_cols)
    importance = importance.sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance.values[:15], y=importance.index[:15])
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"))
    plt.close()

# ===========================
# 5️⃣ Binned Confusion Matrix
# ===========================
print("🔹 Creating binned confusion matrix...")
n_bins = 6
binner = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
y_test_binned = binner.fit_transform(y_test.values.reshape(-1, 1)).astype(int).flatten()
y_pred_binned = binner.transform(y_pred.reshape(-1, 1)).astype(int).flatten()

cm = confusion_matrix(y_test_binned, y_pred_binned)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Binned Sales Ranges)")
plt.xlabel("Predicted Bin")
plt.ylabel("Actual Bin")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix_binned.png"))
plt.close()

# ===========================
# 6️⃣ Model Comparison (All trained models)
# ===========================
print("🔹 Comparing all trained models...")
comparison_metrics = []

for file in os.listdir(ALL_MODELS_DIR):
    if file.endswith(".pkl") and file != "scaler.pkl":
        model_name = file.replace(".pkl", "")
        try:
            mdl = joblib.load(os.path.join(ALL_MODELS_DIR, file))
            preds = mdl.predict(X_test_scaled)
            comparison_metrics.append({
                "Model": model_name,
                "MAE": mean_absolute_error(y_test, preds),
                "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
                "R2": r2_score(y_test, preds)
            })
        except Exception as e:
            print(f"⚠️ Skipping {file}: {e}")

if comparison_metrics:
    comp_df = pd.DataFrame(comparison_metrics)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=comp_df, x="Model", y="R2")
    plt.title("Model Comparison (R² Scores)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison_r2.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=comp_df, x="Model", y="MAE")
    plt.title("Model Comparison (MAE)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison_mae.png"))
    plt.close()

print("✅ All evaluation plots saved in:", PLOTS_DIR)
print("🎯 Final model evaluation complete!")
