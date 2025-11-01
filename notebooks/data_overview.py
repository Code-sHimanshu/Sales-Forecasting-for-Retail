import pandas as pd

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
features = pd.read_csv("data/features.csv")
sample_submission = pd.read_csv("data/sampleSubmission.csv")

# Display basic info
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Features shape:", features.shape)
print("Sample Submission shape:", sample_submission.shape)

# Preview
print(train.head())
print(test.head())
print(features.head())
print(sample_submission.head())


# Check missing values
for name, df in zip(['train', 'test', 'features', 'sample_submission'],
                    [train, test, features, sample_submission]):
    print(f"\n{name.upper()} Missing Values:")
    print(df.isnull().sum())
