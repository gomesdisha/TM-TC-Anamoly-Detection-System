"""
=========================================================
11_live_gru_detection.py

Reads live_tm.csv continuously
Creates rolling window of 60 telemetry samples
Runs trained GRU Autoencoder
Prints anomaly status
Saves anomalies

Author : Disha Gomes
=========================================================
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import deque

# =========================================================
# CONFIG
# =========================================================
# 🔹 Changed to ./live_tm.csv to match where the websocket script saves data
LIVE_FILE = "./live_tm.csv"
MODEL_PATH = "../models/gru_autoencoder.pth"
SCALER_PATH = "../models/scaler.pkl"
THRESHOLD_PATH = "../models/threshold.txt"
FEATURE_LIST = "../config/selected_features.csv"
OUTPUT_FILE = "../reports/live_anomalies.csv"
ANOMALY_FILE = "../reports/results.csv"

SEQ_LEN = 60
INPUT_SIZE = 139
HIDDEN_SIZE = 64
CHECK_INTERVAL = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# MODEL
# =========================================================
class GRUAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.GRU(INPUT_SIZE, HIDDEN_SIZE, batch_first=True)
        self.decoder = nn.GRU(HIDDEN_SIZE, INPUT_SIZE, batch_first=True)

    def forward(self, x):
        _, hidden = self.encoder(x)
        hidden_seq = hidden.repeat(x.size(1), 1, 1).permute(1, 0, 2)
        output, _ = self.decoder(hidden_seq)
        return output

print("Loading model...")
model = GRUAutoencoder()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("Model Loaded")

print("Loading scaler...")
scaler = joblib.load(SCALER_PATH)
print("Scaler Loaded")

print("Loading threshold...")
with open(THRESHOLD_PATH, "r") as f:
    THRESHOLD = float(f.read().strip())
print("Threshold =", THRESHOLD)

# =========================================================
# LOAD FEATURE LIST
# =========================================================
selected_features = pd.read_csv(FEATURE_LIST)
feature_names = selected_features.iloc[:, 0].tolist()
print("Selected Features:", len(feature_names))

# =========================================================
# BUFFER & STATE
# =========================================================
buffer = deque(maxlen=SEQ_LEN)
last_processed_row = 0

# Initialize output file if it doesn't exist
if not os.path.exists(OUTPUT_FILE):
    pd.DataFrame(columns=["Timestamp", "Error", "Status"]).to_csv(OUTPUT_FILE, index=False)

print(f"Monitoring: {LIVE_FILE}")

# =========================================================
# MAIN LOOP
# =========================================================
while True:
    try:
        # Wait until websocket creates file
        if not os.path.exists(LIVE_FILE):
            print("⏳ Waiting for live_tm.csv...")
            time.sleep(CHECK_INTERVAL)
            continue

        live_df = pd.read_csv(LIVE_FILE)

        # Nothing new
        if len(live_df) <= last_processed_row:
            time.sleep(CHECK_INTERVAL)
            continue

        # Process ONLY newly added rows
        new_rows = live_df.iloc[last_processed_row:]
        last_processed_row = len(live_df)

        for _, row in new_rows.iterrows():
            timestamp = row["Timestamp"]

            # Remove timestamp before scaling
            row = row.drop("Timestamp", errors="ignore")

            # Force same order as training
            row = row.reindex(feature_names, fill_value=np.nan)

            # Handle missing values
            row = row.ffill().fillna(0)

            # Scale
            scaled = scaler.transform(pd.DataFrame([row]))[0]

            # Add into rolling buffer
            buffer.append(scaled)
            print(f"Collected {len(buffer)}/{SEQ_LEN}")

            # Wait until 60 samples collected
            if len(buffer) < SEQ_LEN:
                continue

            # ==========================================
            # CREATE SEQUENCE
            # ==========================================
            sequence = np.array(buffer, dtype=np.float32)
            sequence = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # ==========================================
            # RUN MODEL
            # ==========================================
            with torch.no_grad():
                reconstruction = model(sequence)
                error = torch.mean((sequence - reconstruction) ** 2).item()

            # ==========================================
            # PRINT RESULTS
            # ==========================================
            print("\n" + "=" * 60)
            print("Timestamp :", timestamp)
            print("Reconstruction Error :", round(error, 6))
            print("Threshold :", round(THRESHOLD, 6))

            if error > THRESHOLD:
                status = "ANOMALY"
                print("\n🚨 ANOMALY DETECTED 🚨")
            else:
                status = "NORMAL"
                print("\n✅ NORMAL")
            print("=" * 60)

            # ==========================================
            # SAVE ANOMALIES ONLY
            # ==========================================
            if status == "ANOMALY":
                pd.DataFrame([{
                    "Timestamp": timestamp,
                    "Error": error,
                    "Threshold": THRESHOLD,
                    "Status": status
                }]).to_csv(
                    ANOMALY_FILE,
                    mode="a",
                    header=False,
                    index=False
                )
                print("✅ Saved anomaly to:", ANOMALY_FILE)

    except Exception as e:
        print(f"⚠️ Loop error: {e}")

    time.sleep(CHECK_INTERVAL)
