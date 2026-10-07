import pandas as pd
import os

# -----------------------------
# 1. File paths
# -----------------------------
input_file = r"datasets\raw\heart+disease\processed.cleveland.data"
output_file = r"datasets\processed\heart_disease_clean.csv"

# -----------------------------
# 2. Column names
# -----------------------------
columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

# -----------------------------
# 3. Read dataset
# -----------------------------
df = pd.read_csv(
    input_file,
    header=None,
    names=columns
)

print("Original shape:", df.shape)

# -----------------------------
# 4. Replace ? with missing value
# -----------------------------
df = df.replace("?", pd.NA)

# -----------------------------
# 5. Convert columns to numbers
# -----------------------------
for column in columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# -----------------------------
# 6. Fill missing values
# -----------------------------
df["ca"] = df["ca"].fillna(df["ca"].median())
df["thal"] = df["thal"].fillna(df["thal"].mode()[0])

# -----------------------------
# 7. Convert target to binary
# -----------------------------
df["target"] = (df["target"] > 0).astype(int)

# -----------------------------
# 8. Create processed folder
# -----------------------------
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# -----------------------------
# 9. Save cleaned dataset
# -----------------------------
df.to_csv(output_file, index=False)

# -----------------------------
# 10. Display results
# -----------------------------
print("Cleaned shape:", df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["target"].value_counts())

print("\nProcessed dataset saved to:")
print(output_file)