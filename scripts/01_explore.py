import pandas as pd
import numpy as np
import sys
import os

# -------------------------------
# CONFIG
# -------------------------------

SAMPLE_ROWS = 5000
MISSING_THRESHOLD = 0.30
LOW_VARIANCE_THRESHOLD = 0.01
CORR_THRESHOLD = 0.95

# -------------------------------
# LOAD SAMPLE
# -------------------------------

csv_file = sys.argv[1]

print(f"\nLoading sample from {csv_file} ...")

df = pd.read_csv(csv_file, nrows=SAMPLE_ROWS)

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

# -------------------------------
# DATA TYPES
# -------------------------------

dtype_report = pd.DataFrame({
    "column": df.columns,
    "dtype": df.dtypes.astype(str)
})

dtype_report.to_csv(
    "report_dtypes.csv",
    index=False
)

# -------------------------------
# NUMERIC / NON NUMERIC
# -------------------------------

numeric_df = df.copy()

for col in numeric_df.columns:
    if col != "Timestamp":
        numeric_df[col] = pd.to_numeric(
            numeric_df[col],
            errors="coerce"
        )

numeric_cols = numeric_df.select_dtypes(
    include=np.number
).columns.tolist()

non_numeric_cols = [
    c for c in df.columns
    if c not in numeric_cols
]

pd.DataFrame({
    "column": non_numeric_cols
}).to_csv(
    "report_non_numeric.csv",
    index=False
)

print(
    f"Numeric columns     : {len(numeric_cols)}"
)

print(
    f"Non numeric columns : {len(non_numeric_cols)}"
)

# -------------------------------
# MISSING VALUES
# -------------------------------

missing_pct = (
    numeric_df[numeric_cols]
    .isna()
    .mean()
)

missing_report = pd.DataFrame({
    "column": missing_pct.index,
    "missing_pct": missing_pct.values
})

missing_report.to_csv(
    "report_missing.csv",
    index=False
)

# -------------------------------
# CONSTANT FEATURES
# -------------------------------

constant_cols = []

for col in numeric_cols:

    unique_count = (
        numeric_df[col]
        .dropna()
        .nunique()
    )

    if unique_count <= 1:
        constant_cols.append(col)

pd.DataFrame({
    "column": constant_cols
}).to_csv(
    "report_constant.csv",
    index=False
)

# -------------------------------
# LOW VARIANCE FEATURES
# -------------------------------

stds = numeric_df[numeric_cols].std()

low_variance_cols = stds[
    stds < LOW_VARIANCE_THRESHOLD
].index.tolist()

pd.DataFrame({
    "column": low_variance_cols,
    "std": stds[low_variance_cols]
}).to_csv(
    "report_low_variance.csv",
    index=False
)

# -------------------------------
# BUILD CANDIDATE SET
# -------------------------------

candidate_cols = []

for col in numeric_cols:

    if col in constant_cols:
        continue

    if col in low_variance_cols:
        continue

    if missing_pct[col] > MISSING_THRESHOLD:
        continue

    candidate_cols.append(col)

print(
    f"\nCandidate features : {len(candidate_cols)}"
)

# -------------------------------
# CORRELATION ANALYSIS
# -------------------------------

corr_df = numeric_df[candidate_cols]

corr = corr_df.corr().abs()

upper = corr.where(
    np.triu(
        np.ones(corr.shape),
        k=1
    ).astype(bool)
)

high_corr_pairs = []

for col in upper.columns:

    related = upper.index[
        upper[col] > CORR_THRESHOLD
    ].tolist()

    for r in related:

        high_corr_pairs.append(
            [r, col, upper.loc[r, col]]
        )

corr_report = pd.DataFrame(
    high_corr_pairs,
    columns=[
        "feature_1",
        "feature_2",
        "correlation"
    ]
)

corr_report.to_csv(
    "report_correlation.csv",
    index=False
)

# -------------------------------
# FINAL FEATURE LIST
# -------------------------------

selected_features = candidate_cols

pd.DataFrame({
    "feature": selected_features
}).to_csv(
    "selected_features.csv",
    index=False
)

# -------------------------------
# SUMMARY
# -------------------------------

print("\n======================")
print("EXPLORATION COMPLETE")
print("======================")

print(
    f"Selected Features : {len(selected_features)}"
)

print(
    f"Constant Features : {len(constant_cols)}"
)

print(
    f"Low Variance      : {len(low_variance_cols)}"
)

print(
    f"Non Numeric       : {len(non_numeric_cols)}"
)

print("\nGenerated Reports:")

print("report_dtypes.csv")
print("report_missing.csv")
print("report_non_numeric.csv")
print("report_constant.csv")
print("report_low_variance.csv")
print("report_correlation.csv")
print("selected_features.csv")