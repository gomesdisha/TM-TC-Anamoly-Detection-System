import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

# ==========================
# CONFIG
# ==========================

DATA_FILE = "../data/train_jun9_10_clean.csv"

SEQ_LEN = 60

INPUT_SIZE = 139
#HIDDEN_SIZE = 32
HIDDEN_SIZE = 64

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["Timestamp"])

scaler = joblib.load(
    "../models/scaler.pkl"
)

X_scaled = scaler.transform(X)

# ==========================
# CREATE SEQUENCES
# ==========================

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

print(
    "Sequences:",
    sequences.shape
)

# ==========================
# MODEL
# ==========================

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

# ==========================
# ERROR CALCULATION
# ==========================

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

# ==========================
# THRESHOLD
# ==========================

mean_error = errors.mean()

std_error = errors.std()

"""threshold = (
    mean_error +
    3 * std_error
)"""

threshold = np.percentile(
    errors,
    99.5
)

"""print("\nMean Error:", mean_error)
print("Std Error :", std_error)
print("Threshold :", threshold)

print(
    np.percentile(errors,
    [50,90,95,99,99.5,99.9])
)"""

print("\nMean Error:", mean_error)
print("Std Error :", std_error)

print(
    "Percentiles:",
    np.percentile(
        errors,
        [50,90,95,99,99.5,99.9]
    )
)
print("Threshold:", threshold)

# ==========================
# SAVE
# ==========================

np.save(
    "../models/train_errors.npy",
    errors
)

with open(
    "../models/threshold.txt",
    "w"
) as f:

    f.write(str(threshold))

print("\nSaved threshold.")