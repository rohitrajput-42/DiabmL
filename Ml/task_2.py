import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("sample.csv")

# Resolving null values #
# print("Before:")
# print(df.isnull().sum())

# df.fillna(df.median(), inplace=True)
# print("\n")

# print("After:")
# print(df.isnull().sum())






# # Resolving Anomalies #
# print("Before:")
# anomalies_data = ["Glucose", "BloodPressure", "BMI"]
# for anom in anomalies_data:
#     _count = (df[anom] == 0).sum()
#     print(f"{anom} has {_count} zeroes")
# print("\n")

# anomalies_data = ["Glucose", "BloodPressure", "BMI"]
# for anom in anomalies_data:
#     median = df[df[anom] != 0][anom].median()
#     df[anom] = df[anom].replace(0, median)

# print("After:")
# anomalies_data = ["Glucose", "BloodPressure", "BMI"]
# for anom in anomalies_data:
#     _count = (df[anom] == 0).sum()
#     print(f"{anom} has {_count} zeroes")

features = ['Glucose', 'BloodPressure', 'BMI', 'Age']

# Set plot styles
sns.set(style="whitegrid")

# 1. Show basic statistics
print("\nBasic Summary Statistics:")
print(df.describe())

print("\nOutcome Value Counts:")
print(df['Outcome'].value_counts())  # How many with/without diabetes

# 2. Plot histograms of all features
df[features].hist(bins=20, figsize=(12, 10), edgecolor='black')
plt.suptitle("Feature Distributions (Histograms)")
plt.tight_layout()
plt.show()

# 3. Boxplots to compare diabetic vs non-diabetic patients
for feature in features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Outcome', y=feature, data=df)
    plt.title(f"{feature} by Diabetes Outcome")
    plt.xlabel("Diabetes (0 = No, 1 = Yes)")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[features + ['Outcome']].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()