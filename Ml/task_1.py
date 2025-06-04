import pandas as pd

# Loading data from sample csv #
df = pd.read_csv("sample.csv")

print("Diabetes data:")
print(df)
print("\n")

# Null values #
print("Null Values:")
print(df.isnull().sum())
print("\n")

# Anomalies #
print("Anomalies:")
anomalies_data = ["Glucose", "BloodPressure", "BMI"]
for anom in anomalies_data:
    _count = (df[anom] == 0).sum()
    print(f"{anom} has {_count} zeroes")
print("\n")

# Statisticcal metrics #
print("Statistical metrics: ")
print(df.describe())
print("\n")