import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (replace 'transactions.csv' with your file)
df = pd.read_csv('transactions.csv')

# Display dataset info
print("Dataset Info:")
print(df.info())
print("\nDataset Shape:", df.shape)

# Display first few rows
print("\nFirst 5 Rows:")
print(df.head())

# Numerical features summary
print("Numerical Features Summary:")
print(df[['Amount', 'Value', 'CountryCode', 'PricingStrategy', 'FraudResult']].describe())

# Categorical features summary
print("\nCategorical Features Summary:")
print(df[['CurrencyCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId']].describe(include='object'))

# Numerical columns
numerical_cols = ['Amount', 'Value', 'CountryCode', 'PricingStrategy', 'FraudResult']

# Plot histograms with KDE
plt.figure(figsize=(12, 8))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 2, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Categorical columns
categorical_cols = ['CurrencyCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId']

# Plot bar plots
plt.figure(figsize=(12, 8))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(3, 2, i)
    sns.countplot(y=df[col], order=df[col].value_counts().index)
    plt.title(f'Distribution of {col}')
    plt.xlabel('Count')
    plt.ylabel(col)
plt.tight_layout()
plt.show()

# Correlation matrix
correlation_matrix = df[numerical_cols].corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Check missing values
print("Missing Values:")
print(df.isnull().sum())

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Plot box plots
plt.figure(figsize=(12, 8))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(y=df[col])
    plt.title(f'Box Plot of {col}')
plt.tight_layout()
plt.show()


# Convert to datetime
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

# Extract hour, day, month
df['Hour'] = df['TransactionStartTime'].dt.hour
df['DayOfWeek'] = df['TransactionStartTime'].dt.day_name()
df['Month'] = df['TransactionStartTime'].dt.month

# Plot transactions by hour
plt.figure(figsize=(10, 6))
sns.countplot(x=df['Hour'])
plt.title('Transactions by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Count')
plt.show()

# Plot transactions by day of week
plt.figure(figsize=(10, 6))
sns.countplot(x=df['DayOfWeek'], order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Transactions by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Count')
plt.show()