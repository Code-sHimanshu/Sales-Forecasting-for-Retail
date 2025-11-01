# notebooks/data_preprocessing.py
import pandas as pd
import os

# Define data directory
data_dir = os.path.join("data")

# Load datasets
train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
features = pd.read_csv(os.path.join(data_dir, 'features.csv'))

print("Data Loaded Successfully ✅")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Features shape: {features.shape}")

# Convert Date columns to datetime
train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])
features['Date'] = pd.to_datetime(features['Date'])
print("\nDate columns converted to datetime ✔")

# Merge datasets on ['Store', 'Date', 'IsHoliday']
train_merged = pd.merge(train, features, on=['Store', 'Date'], how='left')
test_merged = pd.merge(test, features, on=['Store', 'Date'], how='left')

print("\nDatasets merged successfully ✅")
print(f"Train merged shape: {train_merged.shape}")
print(f"Test merged shape: {test_merged.shape}")

# Combine IsHoliday columns (both from train & features)
train_merged['IsHoliday'] = train_merged['IsHoliday_x'] | train_merged['IsHoliday_y']
test_merged['IsHoliday'] = test_merged['IsHoliday_x'] | test_merged['IsHoliday_y']

# Drop duplicate columns
train_merged.drop(['IsHoliday_x', 'IsHoliday_y'], axis=1, inplace=True)
test_merged.drop(['IsHoliday_x', 'IsHoliday_y'], axis=1, inplace=True)

# Handle missing values (basic cleanup)
print("\nMissing values before cleaning:")
print(train_merged.isnull().sum())

# Fill missing values with 0 for markdowns
markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
for col in markdown_cols:
    if col in train_merged.columns:
        train_merged[col] = train_merged[col].fillna(0)
        test_merged[col] = test_merged[col].fillna(0)

print("\nMissing values handled ✅")

# Save preprocessed data
output_dir = os.path.join("data", "processed")
os.makedirs(output_dir, exist_ok=True)
train_merged.to_csv(os.path.join(output_dir, 'train_preprocessed.csv'), index=False)
test_merged.to_csv(os.path.join(output_dir, 'test_preprocessed.csv'), index=False)

print("\nPreprocessed files saved successfully at 'data/processed/' ✅")
