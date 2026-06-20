import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib


# ==========================
# CONFIG
# ==========================

DATA_FILE = "../data/train_jun9_10_clean.csv"

#SEQ_LEN = 60
BATCH_SIZE = 64
#EPOCHS = 10

INPUT_SIZE = 139
#HIDDEN_SIZE = 32

SEQ_LEN = 60
HIDDEN_SIZE = 64
EPOCHS = 25
BATCH_SIZE = 64

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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

split_idx = int(len(X_scaled) * 0.8)

train_data = X_scaled[:split_idx]
val_data = X_scaled[split_idx:]

print("Train:", train_data.shape)
print("Val  :", val_data.shape)

print("Data Shape:", X_scaled.shape)

# ==========================
# DATASET
# ==========================

class TelemetryDataset(Dataset):

    def __init__(self, data, seq_len):

        self.data = data
        self.seq_len = seq_len

    def __len__(self):

        return len(self.data) - self.seq_len

    def __getitem__(self, idx):

        seq = self.data[
            idx : idx + self.seq_len
        ]

        return torch.tensor(
            seq,
            dtype=torch.float32
        )

"""dataset = TelemetryDataset(
    X_scaled,
    SEQ_LEN
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)"""

train_dataset = TelemetryDataset(
    train_data,
    SEQ_LEN
)

val_dataset = TelemetryDataset(
    val_data,
    SEQ_LEN
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================
# MODEL
# ==========================

class GRUAutoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        """self.encoder = nn.GRU(
            INPUT_SIZE,
            HIDDEN_SIZE,
            batch_first=True
        )

        self.decoder = nn.GRU(
            HIDDEN_SIZE,
            INPUT_SIZE,
            batch_first=True
        )"""

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

model = GRUAutoencoder().to(DEVICE)

criterion = nn.MSELoss()

"""optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)"""

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005
)

# ==========================
# TRAIN
# ==========================

"""print("\nTraining...")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for batch in loader:

        batch = batch.to(DEVICE)

        output = model(batch)

        loss = criterion(
            output,
            batch
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = (
        total_loss /
        len(loader)
    )

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"Loss: {avg_loss:.6f}"
    )"""
best_val_loss = np.inf

print("\nTraining...")

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0

    for batch in train_loader:

        batch = batch.to(DEVICE)

        output = model(batch)

        loss = criterion(output, batch)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for batch in val_loader:

            batch = batch.to(DEVICE)

            output = model(batch)

            loss = criterion(output, batch)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train={train_loss:.6f} | "
        f"Val={val_loss:.6f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "../models/gru_autoencoder.pth"
        )

        print("Saved Best Model")

# ==========================
# SAVE
# ==========================

torch.save(
    model.state_dict(),
    "../models/gru_autoencoder.pth"
)

print("\nModel Saved")