import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report
)

# ============================================================
# PATHS
# ============================================================
MODEL_PATH = "models/best_model.joblib"
SCALER_PATH = "models/scaler.joblib"
TEST_PATH = "data/processed/test_features.csv"
REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)
print("🔹 Loading test data and model...")

# ============================================================
# LOAD DATA & MODEL
# ============================================================
test_data = pd.read_csv(TEST_PATH)
scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)

target_col = 'Weekly_Sales'
has_target = target_col in test_data.columns

if has_target:
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]
else:
    X_test = test_data
    y_test = None

# Scale
# Match test columns with scaler’s training feature names
if hasattr(scaler, 'feature_names_in_'):
    common_features = [col for col in scaler.feature_names_in_ if col in X_test.columns]
    X_test = X_test[common_features]
    print(f"✅ Aligned test features with scaler: {len(common_features)} columns used.")
else:
    print("⚠️ Scaler missing feature name tracking. Using all numeric columns.")
    X_test = X_test.select_dtypes(include=np.number)

# Now scale safely
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

# Save Predictions
pred_df = pd.DataFrame({'Predicted_Weekly_Sales': y_pred})
if has_target:
    pred_df['Actual_Weekly_Sales'] = y_test.values
pred_df.to_csv(os.path.join(REPORT_DIR, "predictions.csv"), index=False)
print("✅ Predictions saved to reports/predictions.csv")

# ============================================================
# EVALUATION SECTION
# ============================================================
if has_target:
    print("\n📊 Evaluating model performance...")

    # Detect if target is numeric (regression) or categorical (classification)
    if np.issubdtype(y_test.dtype, np.number):
        # ========== REGRESSION METRICS ==========
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"MAE:  {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²:   {r2:.2f}")

        # Save metrics
        metrics_report = {"MAE": mae, "RMSE": rmse, "R2": r2}
        pd.DataFrame([metrics_report]).to_json(os.path.join(REPORT_DIR, "evaluation_report.json"), indent=4)
        print("✅ Regression evaluation report saved.")

        # ---------- PLOTS ----------
        plt.figure(figsize=(7, 5))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
        plt.xlabel("Actual Weekly Sales")
        plt.ylabel("Predicted Weekly Sales")
        plt.title("Predicted vs Actual Weekly Sales")
        plt.savefig(os.path.join(REPORT_DIR, "pred_vs_actual.png"))
        plt.close()

        plt.figure(figsize=(7, 5))
        sns.histplot(y_test - y_pred, kde=True, bins=30)
        plt.xlabel("Residuals (Actual - Predicted)")
        plt.title("Residual Distribution")
        plt.savefig(os.path.join(REPORT_DIR, "residuals.png"))
        plt.close()

        print("✅ Regression plots saved to reports folder.")

    else:
        # ========== CLASSIFICATION METRICS ==========
        y_pred_labels = np.round(y_pred) if y_pred.dtype == float else y_pred
        cm = confusion_matrix(y_test, y_pred_labels)
        report = classification_report(y_test, y_pred_labels, output_dict=True)
        print("\nClassification Report:\n", pd.DataFrame(report).transpose())

        # Save report
        pd.DataFrame(report).to_json(os.path.join(REPORT_DIR, "classification_report.json"), indent=4)
        print("✅ Classification report saved.")

        # ---------- CONFUSION MATRIX ----------
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"))
        plt.close()

        print("✅ Confusion matrix saved to reports folder.")

else:
    print("⚠️ 'Weekly_Sales' not found in test data. Only predictions were generated.")
