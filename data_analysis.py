import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("diabetes.csv")

print(df.head())

print("\ndataset shape")
print(df.shape)

print("\ndataset info")
print(df.info())

print("\nstatistical summary")
print(df.describe())

print("\nmissing values")
print(df.isnull().sum())

columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

print("\nZero Values Before Cleaning")

for col in columns:
    print(col,(df[col]==0).sum())

for col in columns:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

print("\nzero values after cleaning")

for col in columns:
    print(col,(df[col]==0).sum())

df.to_csv("cleaned_diabetes.csv", index=False)

print("\nCorrelation Matrix")
print(df.corr())


plt.figure(figsize=(6,4))
sns.countplot(x="Outcome", data=df)
plt.title("Diabetes Outcome Distribution")
plt.savefig("outcome_distribution.png")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),
            annot=True,
            cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df["Glucose"],
             kde=True)

plt.title("Glucose Distribution")
plt.savefig("glucose_distribution.png")
plt.show()