import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

# =========================
# CONFIG
# =========================

DATA_FILE = "../data/test_jun11_injected.csv"

SEQ_LEN = 60

INPUT_SIZE = 139
#HIDDEN_SIZE = 32
HIDDEN_SIZE = 64

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# =========================
# LOAD THRESHOLD
# =========================

# Use 99.5 percentile threshold

#THRESHOLD = 3.485
with open(
    "../models/threshold.txt",
    "r"
) as f:

    THRESHOLD = float(
        f.read().strip()
    )

print("Threshold:", THRESHOLD)


# =========================
# LOAD DATA
# =========================

print("Loading June 10 telemetry...")

df = pd.read_csv(DATA_FILE)

timestamps = df["Timestamp"].copy()

X = df.drop(
    columns=["Timestamp"]
)

print(X.shape)

# =========================
# SCALE
# =========================

scaler = joblib.load(
    "../models/scaler.pkl"
)

X_scaled = scaler.transform(X)

# =========================
# CREATE SEQUENCES
# =========================

sequences = []

sequence_times = []

for i in range(
        len(X_scaled) - SEQ_LEN):

    sequences.append(
        X_scaled[i:i+SEQ_LEN]
    )

    sequence_times.append(
        timestamps.iloc[i+SEQ_LEN]
    )

sequences = np.array(
    sequences,
    dtype=np.float32
)

print(
    "Sequences:",
    sequences.shape
)

# =========================
# MODEL
# =========================

class GRUAutoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.GRU(
            INPUT_SIZE,
            HIDDEN_SIZE,
            batch_first=True
        )

        self.decoder = nn.GRU(
            HIDDEN_SIZE,
            INPUT_SIZE,
            batch_first=True
        )

    def forward(self, x):

        _, hidden = self.encoder(x)

        hidden_seq = hidden.repeat(
            x.size(1),
            1,
            1
        ).permute(1,0,2)

        output, _ = self.decoder(
            hidden_seq
        )

        return output

model = GRUAutoencoder()

model.load_state_dict(
    torch.load(
        "../models/gru_autoencoder.pth",
        map_location=DEVICE
    )
)

model.to(DEVICE)

model.eval()

# =========================
# ANOMALY DETECTION
# =========================

errors = []

with torch.no_grad():

    for seq in sequences:

        seq = torch.tensor(
            seq,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        recon = model(seq)

        error = torch.mean(
            (seq - recon) ** 2
        ).item()

        errors.append(error)

errors = np.array(errors)

# =========================
# ANOMALIES
# =========================

anomaly_mask = (
    errors > THRESHOLD
)

anomaly_count = (
    anomaly_mask.sum()
)

print("\nResults")
print("Total Sequences :", len(errors))
print("Anomalies       :", anomaly_count)

print(
    "Anomaly %       :",
    round(
        anomaly_count /
        len(errors)
        * 100,
        3
    )
)

print(
    np.percentile(
        errors,
        [50,90,95,99,99.5,99.9]
    )
)

#SAVE REPORT

"""results = pd.DataFrame({

    "Timestamp":
        sequence_times,

    "Error":
        errors,

    "Anomaly":
        anomaly_mask
})

results.to_csv(
    "../reports/june10_anomalies.csv",
    index=False
)

print(
    "\nSaved:"
)

print(
    "../reports/june10_anomalies.csv"
)"""

# =========================
# SAVE REPORTS
# =========================

results = pd.DataFrame({

    "Timestamp": sequence_times,

    "Error": errors,

    "Anomaly": anomaly_mask
})

# -------------------------
# Save full report
# -------------------------

"""results.to_csv(
    "../reports/june11_all_results.csv",
    index=False
)"""

# -------------------------
# Save anomalies only
# -------------------------

anomalies_only = results[
    results["Anomaly"] == True
]

"""anomalies_only.to_csv(
    "../reports/june11_anomalies_only.csv",
    index=False
)"""

results.to_csv(
    "../reports/june11_injected_results.csv",
    index=False
)

print(" ADDED INJECTED ANAMOLIES TO CSV !! ")

# -------------------------
# Print anomalies
# -------------------------

print("\n")
print("=" * 50)
print("FIRST 50 ANOMALIES DETECTED")
print("=" * 50)

"""for _, row in anomalies_only.iterrows():

    print(
        f"{row['Timestamp']} | "
        f"Error={row['Error']:.4f}"
    )"""

for _, row in anomalies_only.head(50).iterrows():

    print(
        f"{row['Timestamp']} | "
        f"Error={row['Error']:.4f}"
    )

print(
    f"\nTotal anomalies saved: {len(anomalies_only)}"
)

print("\nSaved:")

print(
    "../reports/june11_all_results.csv"
)

print(
    "../reports/june11_anomalies_only.csv"
)