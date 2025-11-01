import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load preprocessed data
train = pd.read_csv("data/processed/train_preprocessed.csv")

print("✅ Data Loaded for EDA")
print("Shape of Train Data:", train.shape)
print(train.head())

# Convert Date to datetime
train['Date'] = pd.to_datetime(train['Date'])

# ---------- 1. Basic Summary ----------
print("\nData Info:")
print(train.info())

print("\nStatistical Summary:")
print(train.describe())

# ---------- 2. Sales Overview ----------
# Total Sales over time
plt.figure(figsize=(12,6))
sales_over_time = train.groupby('Date')['Weekly_Sales'].sum().reset_index()
plt.plot(sales_over_time['Date'], sales_over_time['Weekly_Sales'])
plt.title("Total Weekly Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.grid(True)
plt.tight_layout()
plt.savefig("reports/sales_trend_over_time.png")
plt.close()

print("📊 Saved plot: reports/sales_trend_over_time.png")

# ---------- 3. Store-level Analysis ----------
store_sales = train.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending=False)
plt.figure(figsize=(10,5))
store_sales.plot(kind='bar')
plt.title("Total Sales by Store")
plt.xlabel("Store")
plt.ylabel("Total Weekly Sales")
plt.tight_layout()
plt.savefig("reports/sales_by_store.png")
plt.close()

print("📊 Saved plot: reports/sales_by_store.png")

# ---------- 4. Department-level Analysis ----------
dept_sales = train.groupby('Dept')['Weekly_Sales'].sum().sort_values(ascending=False)
plt.figure(figsize=(10,5))
dept_sales.head(20).plot(kind='bar')
plt.title("Top 20 Departments by Sales")
plt.xlabel("Department")
plt.ylabel("Total Weekly Sales")
plt.tight_layout()
plt.savefig("reports/top_20_departments.png")
plt.close()

print("📊 Saved plot: reports/top_20_departments.png")

# ---------- 5. Holiday vs Non-Holiday Sales ----------
holiday_sales = train.groupby('IsHoliday')['Weekly_Sales'].mean().reset_index()
sns.barplot(data=holiday_sales, x='IsHoliday', y='Weekly_Sales')
plt.title("Average Sales on Holidays vs Non-Holidays")
plt.xlabel("IsHoliday")
plt.ylabel("Average Weekly Sales")
plt.tight_layout()
plt.savefig("reports/holiday_vs_nonholiday_sales.png")
plt.close()

print("📊 Saved plot: reports/holiday_vs_nonholiday_sales.png")

# ---------- 6. Correlation Heatmap ----------
numeric_features = train.select_dtypes(include=['float64', 'int64'])
corr = numeric_features.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.savefig("reports/correlation_heatmap.png")
plt.close()

print("📊 Saved plot: reports/correlation_heatmap.png")

print("\nStep 3 (EDA) completed successfully 🚀")
