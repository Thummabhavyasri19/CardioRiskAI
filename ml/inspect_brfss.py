import pandas as pd


# ---------------------------------------
# BRFSS file
# ---------------------------------------
file_path = r"datasets\raw\brfss_2025\LLCP2025.XPT"


# ---------------------------------------
# Variables for Basic Screening
# ---------------------------------------
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


print("Reading BRFSS data in chunks...")


reader = pd.read_sas(
    file_path,
    format="xport",
    iterator=True,
    chunksize=10000
)


chunks = []


for chunk_number, chunk in enumerate(reader, start=1):

    selected = chunk[selected_columns].copy()

    chunks.append(selected)

    print(
        f"Processed chunk {chunk_number} "
        f"with {len(selected)} records"
    )


df = pd.concat(
    chunks,
    ignore_index=True
)


print("\n=======================================")
print("DATASET INFORMATION")
print("=======================================")

print("Shape:", df.shape)


print("\n=======================================")
print("MISSING VALUES")
print("=======================================")

print(df.isna().sum())


print("\n=======================================")
print("UNIQUE VALUES")
print("=======================================")

for column in selected_columns:

    print("\n---------------------------------------")
    print(column)
    print("---------------------------------------")

    print(
        df[column]
        .value_counts(dropna=False)
        .sort_index()
    )


print("\n=======================================")
print("TARGET DISTRIBUTION")
print("=======================================")

print(
    df["_MICHD"]
    .value_counts(dropna=False)
)


print("\nInspection completed.")