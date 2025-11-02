# notebooks/feature_engineering.py

import pandas as pd
import numpy as np
import os

# Define paths
BASE_PATH = "C:/Users/Himanshu Singh/projects/InlighnX_Internship_Projects/Sales_Forecasting_For_Retail/data"
TRAIN_PATH = os.path.join(BASE_PATH, "train.csv")
TEST_PATH = os.path.join(BASE_PATH, "test.csv")
FEATURES_PATH = os.path.join(BASE_PATH, "features.csv")

# Load datasets
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
features = pd.read_csv(FEATURES_PATH)

print("Data Loaded Successfully ✅")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Features shape: {features.shape}")

# Convert Date columns
for df in [train, test, features]:
    df['Date'] = pd.to_datetime(df['Date'])

# Merge datasets
train_merged = pd.merge(train, features, on=['Store', 'Date'], how='left')
test_merged = pd.merge(test, features, on=['Store', 'Date'], how='left')

print("\nDatasets merged successfully ✅")
print(f"Train merged shape: {train_merged.shape}")
print(f"Test merged shape: {test_merged.shape}")

# Handle duplicate IsHoliday columns
holiday_col_train = 'IsHoliday_x' if 'IsHoliday_x' in train_merged.columns else 'IsHoliday'
holiday_col_test = 'IsHoliday_x' if 'IsHoliday_x' in test_merged.columns else 'IsHoliday'

train_merged['IsHoliday'] = train_merged[holiday_col_train].astype(bool)
test_merged['IsHoliday'] = test_merged[holiday_col_test].astype(bool)

# Feature Engineering
for df in [train_merged, test_merged]:
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)

# Aggregate historical features
sales_by_store = train_merged.groupby('Store')['Weekly_Sales'].mean().reset_index()
sales_by_store.columns = ['Store', 'Avg_Store_Sales']

train_merged = train_merged.merge(sales_by_store, on='Store', how='left')
test_merged = test_merged.merge(sales_by_store, on='Store', how='left')

print("\nFeature engineering completed successfully ✅")

# Save processed files
OUTPUT_PATH = os.path.join(BASE_PATH, "processed")
os.makedirs(OUTPUT_PATH, exist_ok=True)

train_merged.to_csv(os.path.join(OUTPUT_PATH, "train_features.csv"), index=False)
test_merged.to_csv(os.path.join(OUTPUT_PATH, "test_features.csv"), index=False)

print(f"\nProcessed files saved successfully to: {OUTPUT_PATH}")
