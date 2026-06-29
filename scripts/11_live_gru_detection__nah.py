"""
11_live_gru_detection.py
-----------------------------------------
Template for live GRU inference using the trained
139-feature GRU autoencoder.

TODO before running:
1. Set MODEL_PATH, SCALER_PATH, THRESHOLD_PATH.
2. Feed LIVE_ROWS from your websocket (09 script).
3. Keep only selected 139 features in the same order.
"""

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import deque

SEQ_LEN = 60
INPUT_SIZE = 139
HIDDEN_SIZE = 64

MODEL_PATH = "../models/gru_autoencoder.pth"
SCALER_PATH = "../models/scaler.pkl"
THRESHOLD_PATH = "../models/threshold.txt"

# ---------- MODEL ----------
class GRUAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.GRU(INPUT_SIZE, HIDDEN_SIZE, batch_first=True)
        self.decoder = nn.GRU(HIDDEN_SIZE, INPUT_SIZE, batch_first=True)

    def forward(self, x):
        _, h = self.encoder(x)
        h = h.repeat(x.size(1),1,1).permute(1,0,2)
        out,_ = self.decoder(h)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUAutoencoder()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

scaler = joblib.load(SCALER_PATH)
with open(THRESHOLD_PATH) as f:
    THRESHOLD=float(f.read().strip())

buffer = deque(maxlen=SEQ_LEN)

def process_live_packet(feature_dict):
    """
    feature_dict must contain:
    Timestamp
    + 139 telemetry features
    """

    ts = feature_dict["Timestamp"]

    row = pd.DataFrame([feature_dict])

    # Remove timestamp before scaling
    X = row.drop(columns=["Timestamp"])

    # Fill missing values
    X = X.fillna(method="ffill", axis=1).fillna(0)

    X_scaled = scaler.transform(X)

    buffer.append(X_scaled[0])

    print(f"Collected {len(buffer)}/{SEQ_LEN}")

    if len(buffer) < SEQ_LEN:
        return

    seq = np.array(buffer, dtype=np.float32)
    seq = torch.tensor(seq).unsqueeze(0).to(device)

    with torch.no_grad():
        recon = model(seq)
        err = torch.mean((seq-recon)**2).item()

    print("="*50)
    print("Timestamp :", ts)
    print("Reconstruction Error :", err)
    print("Threshold :", THRESHOLD)

    if err > THRESHOLD:
        print("🚨 ANOMALY DETECTED")
    else:
        print("✅ NORMAL")
    print("="*50)

"""
Usage:

From your websocket:

telemetry["Timestamp"]=payload["frame_time"]
process_live_packet(telemetry)

Alternative:
If websocket gives a batch instead of one row,
iterate through each row and call process_live_packet().
"""
