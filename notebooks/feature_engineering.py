import pandas as pd
import os

# Set paths
BASE_PATH = os.path.join(os.getcwd(), 'data')
TRAIN_PATH = os.path.join(BASE_PATH, 'train', 'train.csv')
TEST_PATH = os.path.join(BASE_PATH, 'test', 'test.csv')
FEATURES_PATH = os.path.join(BASE_PATH, 'features', 'features.csv')

# Load data
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
features = pd.read_csv(FEATURES_PATH)

# Merge features with train and test sets
train_merged = train.merge(features, on=['store', 'item'], how='left')
test_merged = test.merge(features, on=['store', 'item'], how='left')

# Convert date to datetime
train_merged['date'] = pd.to_datetime(train_merged['date'])
test_merged['date'] = pd.to_datetime(test_merged['date'])

# Create time-based features
for df in [train_merged, test_merged]:
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

# Handle missing values
train_merged.fillna(0, inplace=True)
test_merged.fillna(0, inplace=True)

# Save processed data
PROCESSED_PATH = os.path.join(BASE_PATH, 'processed')
os.makedirs(PROCESSED_PATH, exist_ok=True)

train_merged.to_csv(os.path.join(PROCESSED_PATH, 'train_processed.csv'), index=False)
test_merged.to_csv(os.path.join(PROCESSED_PATH, 'test_processed.csv'), index=False)

print("✅ Feature engineering complete.")
print(f"Train shape after processing: {train_merged.shape}")
print(f"Test shape after processing: {test_merged.shape}")
