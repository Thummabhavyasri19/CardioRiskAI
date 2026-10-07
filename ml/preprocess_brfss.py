import pandas as pd
import os


# =======================================
# File paths
# =======================================

input_file = r"datasets\raw\brfss_2025\LLCP2025.XPT"

output_file = r"datasets\processed\basic_screening_clean.csv"


# =======================================
# Variables we need
# =======================================

selected_columns = [
    "_AGEG5YR",
    "_SEX",
    "_BMI5",
    "_RFSMOK3",
    "_TOTINDA",
    "_RFHYPE6",
    "DIABETE4",
    "GENHLTH",
    "SLEPTIM1",
    "DRNKANY6",
    "CVDSTRK3",
    "_MICHD"
]


# =======================================
# Process BRFSS in chunks
# =======================================

print("Reading BRFSS dataset...")

reader = pd.read_sas(
    input_file,
    format="xport",
    iterator=True,
    chunksize=10000
)


processed_chunks = []

chunk_number = 0


for chunk in reader:

    chunk_number += 1

    # ---------------------------------------
    # Keep only required variables
    # ---------------------------------------
    df = chunk[selected_columns].copy()

    # ---------------------------------------
    # Convert columns to numeric
    # ---------------------------------------
    for column in selected_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # =======================================
    # Remove invalid codes
    # =======================================

    # Age
    df.loc[
        ~df["_AGEG5YR"].between(1, 13),
        "_AGEG5YR"
    ] = pd.NA

    # Sex
    df.loc[
        ~df["_SEX"].isin([1, 2]),
        "_SEX"
    ] = pd.NA

    # BMI
    # BRFSS stores BMI multiplied by 100
    df["_BMI5"] = df["_BMI5"] / 100

    df.loc[
        ~df["_BMI5"].between(10, 100),
        "_BMI5"
    ] = pd.NA

    # Smoking
    df.loc[
        ~df["_RFSMOK3"].isin([1, 2]),
        "_RFSMOK3"
    ] = pd.NA

    # Physical inactivity
    df.loc[
        ~df["_TOTINDA"].isin([1, 2]),
        "_TOTINDA"
    ] = pd.NA

    # High blood pressure history
    df.loc[
        ~df["_RFHYPE6"].isin([1, 2]),
        "_RFHYPE6"
    ] = pd.NA

    # Diabetes status
    df.loc[
        ~df["DIABETE4"].isin([1, 2, 3, 4]),
        "DIABETE4"
    ] = pd.NA

    # General health
    df.loc[
        ~df["GENHLTH"].isin([1, 2, 3, 4, 5]),
        "GENHLTH"
    ] = pd.NA

    # Sleep
    df.loc[
        ~df["SLEPTIM1"].between(1, 24),
        "SLEPTIM1"
    ] = pd.NA

    # Alcohol
    df.loc[
        ~df["DRNKANY6"].isin([1, 2]),
        "DRNKANY6"
    ] = pd.NA

    # Previous stroke
    df.loc[
        ~df["CVDSTRK3"].isin([1, 2]),
        "CVDSTRK3"
    ] = pd.NA

    # Target
    df.loc[
        ~df["_MICHD"].isin([1, 2]),
        "_MICHD"
    ] = pd.NA

    # ---------------------------------------
    # Remove rows with missing values
    # ---------------------------------------
    df = df.dropna()

    # ---------------------------------------
    # Convert values into model-friendly form
    # ---------------------------------------

    # Age group
    df["age_group"] = df["_AGEG5YR"].astype(int)

    # Sex:
    # 1 = Male
    # 2 = Female
    df["sex"] = (
        df["_SEX"]
        .map({
            1: 0,
            2: 1
        })
        .astype(int)
    )

    # BMI
    df["bmi"] = df["_BMI5"].astype(float)

    # Current smoking:
    # 1 = Current smoker
    # 2 = Not current smoker
    df["smoking"] = (
        df["_RFSMOK3"]
        .map({
            1: 1,
            2: 0
        })
        .astype(int)
    )

    # Physical inactivity:
    # _TOTINDA:
    # 1 = Active
    # 2 = Inactive
    df["physical_inactivity"] = (
        df["_TOTINDA"]
        .map({
            1: 0,
            2: 1
        })
        .astype(int)
    )

    # High blood pressure history:
    # 1 = Yes
    # 2 = No
    df["high_blood_pressure"] = (
        df["_RFHYPE6"]
        .map({
            1: 1,
            2: 0
        })
        .astype(int)
    )

    # Diabetes status:
    # Keep original four meaningful categories:
    # 1 = Diabetes
    # 2 = Gestational diabetes
    # 3 = No diabetes
    # 4 = Prediabetes
    df["diabetes_status"] = (
        df["DIABETE4"]
        .astype(int)
    )

    # General health:
    # 1 = Excellent
    # 2 = Very good
    # 3 = Good
    # 4 = Fair
    # 5 = Poor
    df["general_health"] = (
        df["GENHLTH"]
        .astype(int)
    )

    # Sleep hours
    df["sleep_hours"] = (
        df["SLEPTIM1"]
        .astype(int)
    )

    # Alcohol use:
    # 1 = Yes
    # 2 = No
    df["alcohol_use"] = (
        df["DRNKANY6"]
        .map({
            1: 1,
            2: 0
        })
        .astype(int)
    )

    # Previous stroke:
    # 1 = Yes
    # 2 = No
    df["previous_stroke"] = (
        df["CVDSTRK3"]
        .map({
            1: 1,
            2: 0
        })
        .astype(int)
    )

    # Target:
    # _MICHD = 1 → positive
    # _MICHD = 2 → negative
    df["target"] = (
        df["_MICHD"]
        .map({
            1: 1,
            2: 0
        })
        .astype(int)
    )

    # ---------------------------------------
    # Keep only final model columns
    # ---------------------------------------

    final_columns = [
        "age_group",
        "sex",
        "bmi",
        "smoking",
        "physical_inactivity",
        "high_blood_pressure",
        "diabetes_status",
        "general_health",
        "sleep_hours",
        "alcohol_use",
        "previous_stroke",
        "target"
    ]

    df = df[final_columns]

    processed_chunks.append(df)

    print(
        f"Processed chunk {chunk_number}: "
        f"{len(df)} valid records"
    )


# =======================================
# Combine all processed chunks
# =======================================

final_df = pd.concat(
    processed_chunks,
    ignore_index=True
)


# =======================================
# Create output directory
# =======================================

os.makedirs(
    os.path.dirname(output_file),
    exist_ok=True
)


# =======================================
# Save dataset
# =======================================

final_df.to_csv(
    output_file,
    index=False
)


# =======================================
# Display results
# =======================================

print("\n=======================================")
print("BASIC SCREENING DATASET")
print("=======================================")

print("Final shape:", final_df.shape)

print("\nMissing values:")
print(final_df.isnull().sum())

print("\nTarget distribution:")
print(final_df["target"].value_counts())

print("\nTarget percentage:")
print(
    final_df["target"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nFirst 5 rows:")
print(final_df.head())

print("\nSaved to:")
print(output_file)