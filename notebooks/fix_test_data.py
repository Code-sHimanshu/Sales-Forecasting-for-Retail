import pandas as pd

# Load preprocessed test data (features only)
test = pd.read_csv("data/processed/test_preprocessed.csv")

# Load original test data that contains Weekly_Sales
# If you split from 'train.csv', load that original dataset
original = pd.read_csv("data/raw/test.csv")  # CHANGE this to your actual raw file

# Merge Weekly_Sales back if possible (based on Store, Dept, Date)
if "Weekly_Sales" in original.columns:
    merged = test.merge(
        original[["Store", "Dept", "Date", "Weekly_Sales"]],
        on=["Store", "Dept", "Date"],
        how="left"
    )
    merged.to_csv("data/processed/test_preprocessed_fixed.csv", index=False)
    print("✅ Fixed file saved as data/processed/test_preprocessed_fixed.csv")
else:
    print("❌ The raw data doesn't contain 'Weekly_Sales' column. Please check your source file.")
