"""import pandas as pd
import numpy as np

selected = pd.read_csv(
    "../config/selected_features.csv"
)

features = selected.iloc[:,0].tolist()

row = {}

for f in features:
    row[f] = telemetry.get(
        f,
        np.nan
    )

df = pd.DataFrame([row])

print(df.shape)
print(df.head())
print(
    df.isna().sum().sum()
)"""
import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD SAVED TELEMETRY CSV
# ==========================================
# Since you're in ~/Aravind/disha-jun22/scripts/, use "./live_tm.csv"
csv_path = "./live_tm.csv"
df = pd.read_csv(csv_path)

print(f"✅ Loaded {len(df)} rows from live_tm.csv")

# ==========================================
# 2. LOAD EXPECTED FEATURES
# ==========================================
features_df = pd.read_csv("../config/selected_features.csv")
features = features_df.iloc[:, 0].tolist()

# ==========================================
# 3. ALIGN COLUMNS & FILL MISSING VALUES
# ==========================================
# If the CSV is missing any of your expected features, add them as NaN
for col in features:
    if col not in df.columns:
        df[col] = np.nan

# Reorder columns to strictly match your expected feature list
df = df[features]

# ==========================================
# 4. PRINT OUTPUTS (Exactly as you requested)
# ==========================================
print(f"\nShape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print(f"\nTotal missing values (NaN): {df.isna().sum().sum()}")

# ==========================================
# 5. (Optional) ROW-BY-ROW PROCESSING
# ==========================================
# If you need to run inference one row at a time:
print("\n--- Row-by-Row Example ---")
for idx, row in df.iterrows():
    # row is already a pandas Series. No need for .get()
    missing_count = row.isna().sum()
    print(f"Row {idx} | Features: {len(row)} | Missing: {missing_count}")

    # 🔹 Put your model prediction here:
    # fv = row.values.reshape(1, -1)  # Convert to numpy array
    # status = detect(fv)            # Your anomaly detection function

"""(ml_env) csrspdev@pdap-system2:~/Aravind/disha-jun22/scripts$ ls
01_explore.py  03_train_prep.py  05_generate_threshold.py  07_root_cause_analysis.py  09_websocket_debug.py  10_live_dataframe_builder.py  checker.txt          live_tm.csv  split_days.py   TESTER.txt
02_clean.py    04_train_gru.py   06_test_june11.py         08_check_pid_mapping.py    09_websock.py          11_live_model_test.py         inject_anomalies.py  results.txt  tester_file.py
this is my current directory, idk why telemetry is underlined in red, how do i make it work, do i change it to live_csv?"""

"""Convert websocket telemetry
into the same format as
train_jun9_10_clean.csv
Expected:

(1,139)"""