import pandas as pd
import numpy as np
import sys
import os

# ----------------------------
# INPUTS
# ----------------------------

TM_FILE = sys.argv[1]

FEATURE_FILE = "../config/selected_features.csv"

# ----------------------------
# CREATE FOLDERS
# ----------------------------

os.makedirs("../reports", exist_ok=True)
os.makedirs("../data", exist_ok=True)

# ----------------------------
# LOAD FEATURE LIST
# ----------------------------

features = pd.read_csv(FEATURE_FILE)

feature_list = features.iloc[:, 0].tolist()

print(f"Selected Features: {len(feature_list)}")

# ----------------------------
# LOAD DATA (CHUNKED)
# ----------------------------

print("Loading telemetry...")

use_cols = ["Timestamp"] + feature_list

chunks = []

total_rows = 0

for chunk in pd.read_csv(
        TM_FILE,
        usecols=use_cols,
        chunksize=1000,
        low_memory=False):

    chunks.append(chunk)

    total_rows += len(chunk)

    print(f"Loaded {total_rows} rows")

df = pd.concat(
    chunks,
    ignore_index=True
)

print("\nLoaded Dataset Shape:")
print(df.shape)

# ----------------------------
# NUMERIC CONVERSION
# ----------------------------

print("\nConverting to numeric...")

for col in feature_list:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# ----------------------------
# INVALID VALUE CLEANING
# ----------------------------

print("Replacing invalid values...")

INVALID_VALUES = [
    -47.86,
    -32768,
    -9999,
    65535,
    -26.59
]

for val in INVALID_VALUES:

    df.replace(
        val,
        np.nan,
        inplace=True
    )

# ----------------------------
# INFINITE VALUES
# ----------------------------

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

# ----------------------------
# MISSING REPORT (BEFORE FILL)
# ----------------------------

missing_before = df.isna().sum()

report_before = pd.DataFrame({
    "feature": missing_before.index,
    "missing_count": missing_before.values
})

report_before.to_csv(
    "../reports/missing_before_fill.csv",
    index=False
)

# ----------------------------
# FILL MISSING VALUES
# ----------------------------

print("Filling missing values...")

df = df.ffill()
df = df.bfill()

# ----------------------------
# DROP REMAINING NaNs
# ----------------------------

df.dropna(inplace=True)

# ----------------------------
# MISSING REPORT (AFTER FILL)
# ----------------------------

missing_after = df.isna().sum()

report_after = pd.DataFrame({
    "feature": missing_after.index,
    "missing_count": missing_after.values
})

report_after.to_csv(
    "../reports/missing_after_fill.csv",
    index=False
)

# ----------------------------
# TIMESTAMP SORT
# ----------------------------

print("Sorting timestamps...")

df.sort_values(
    "Timestamp",
    inplace=True
)

df.drop_duplicates(
    subset="Timestamp",
    inplace=True
)

# ----------------------------
# SAVE CLEAN DATA
# ----------------------------

filename = os.path.basename(TM_FILE)

name = filename.replace(".csv", "")

output_file = f"../data/{name}_clean.csv"

df.to_csv(
    output_file,
    index=False
)

# ----------------------------
# SUMMARY
# ----------------------------

print("\n======================")
print("CLEANING COMPLETE")
print("======================")

print("Final Shape:")
print(df.shape)

print("\nOutput File:")
print(output_file)

print("\nReports Generated:")
print("../reports/missing_before_fill.csv")
print("../reports/missing_after_fill.csv")