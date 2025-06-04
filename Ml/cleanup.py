import pandas as pd

df = pd.read_csv("sample.csv")

columns_to_fix = ['Glucose', 'BloodPressure', 'BMI']
for col in columns_to_fix:
    median = df[df[col] != 0][col].median()
    df[col] = df[col].replace(0, median)

df.fillna(df.median(numeric_only=True), inplace=True)

df.to_csv("sample_cleaned.csv", index=False)