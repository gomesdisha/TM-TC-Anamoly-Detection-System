import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

# =========================
# CONFIG
# =========================

DATA_FILE = "../data/test_jun11_clean.csv"

SEQ_LEN = 60

INPUT_SIZE = 139
HIDDEN_SIZE = 64

TOP_K = 15

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(DATA_FILE)

timestamps = df["Timestamp"]

feature_names = list(
    df.drop(columns=["Timestamp"]).columns
)

X = df.drop(
    columns=["Timestamp"]
)

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

for i in range(
        len(X_scaled) - SEQ_LEN):

    sequences.append(
        X_scaled[i:i+SEQ_LEN]
    )

sequences = np.array(
    sequences,
    dtype=np.float32
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
# FEATURE ERRORS
# =========================

feature_errors = np.zeros(INPUT_SIZE)

count = 0

with torch.no_grad():

    for seq in sequences:

        seq = torch.tensor(
            seq,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        recon = model(seq)

        per_feature_error = (
            (seq - recon) ** 2
        ).mean(
            dim=(0,1)
        )

        feature_errors += (
            per_feature_error
            .cpu()
            .numpy()
        )

        count += 1

feature_errors /= count

# =========================
# REPORT
# =========================

report = pd.DataFrame({

    "Feature": feature_names,

    "Mean_Reconstruction_Error":
        feature_errors

})

report = report.sort_values(
    "Mean_Reconstruction_Error",
    ascending=False
)

print(
    "\nTop Contributors:\n"
)

print(
    report.head(TOP_K)
)

report.to_csv(
    "../reports/root_cause_features.csv",
    index=False
)

print(
    "\nSaved:"
)

print(
    "../reports/root_cause_features.csv"
)