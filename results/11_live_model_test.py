"""mport joblib
import torch
import pandas as pd
import numpy as np

scaler = joblib.load("../models/scaler.pkl")

df = pd.read_csv(
    "../data/sample_live_row.csv"
)

X = scaler.transform(df)

print(X.shape)

print(np.isnan(X).sum())
"""
import joblib
import pandas as pd
import numpy as np

# 1. Load scaler
scaler = joblib.load("../models/scaler.pkl")

# 2. Load expected feature order (must match training data exactly)
features = pd.read_csv("../config/selected_features.csv").iloc[:, 0].tolist()

# 3. Load live telemetry & align columns to training order
df = pd.read_csv("./live_tm.csv")
df = df.reindex(columns=features, fill_value=0)  # ⚠️ Fills missing/NaN columns with 0

# 4. Scale
X = scaler.transform(df)

print(f"Shape: {X.shape}")
print(f"NaN count after scaling: {np.isnan(X).sum()}")

# Expected: (1, 139)

"""Can my trained model
accept live telemetry?

Expected:

(1,139)"""

"""selected_features.csv

gru_autoencoder.pth

scaler.pkl

threshold.txt

04_train_gru.py

05_generate_threshold.py

06_test_june11.py

07_root_cause_analysis.py"""