import pandas as pd
import numpy as np
import joblib
import os
import sys

from sklearn.preprocessing import StandardScaler

# ==================================
# CONFIG
# ==================================

CSV_FILE = sys.argv[1]

SEQ_LEN = 120

# ==================================
# LOAD
# ==================================

print("Loading cleaned telemetry...")

df = pd.read_csv(CSV_FILE)

print(df.shape)

# ==================================
# REMOVE TIMESTAMP
# ==================================

timestamps = df["Timestamp"]

X = df.drop(columns=["Timestamp"])

# ==================================
# TRAIN / VALIDATION SPLIT
# ==================================

split_idx = int(len(X) * 0.8)

X_train = X.iloc[:split_idx]

X_val = X.iloc[split_idx:]

print("\nTrain Shape:", X_train.shape)
print("Validation Shape:", X_val.shape)

# ==================================
# SCALE
# ==================================

print("\nScaling...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)

os.makedirs("../models", exist_ok=True)

joblib.dump(
    scaler,
    "../models/scaler.pkl"
)

print("Scaler saved.")

"""# ==================================
# WINDOW CREATION
# ==================================

def create_sequences(data, seq_len):

    sequences = []

    for i in range(
            len(data) - seq_len):

        sequences.append(
            data[i:i + seq_len]
        )

    return np.array(sequences)

print("\nCreating sequences...")

X_train_seq = create_sequences(
    X_train_scaled,
    SEQ_LEN
)

X_val_seq = create_sequences(
    X_val_scaled,
    SEQ_LEN
)

print(
    "Train Sequences:",
    X_train_seq.shape
)

print(
    "Validation Sequences:",
    X_val_seq.shape
)

# ==================================
# SAVE
# ==================================

np.save(
    "../data/X_train.npy",
    X_train_seq
)

np.save(
    "../data/X_val.npy",
    X_val_seq
)

print("\nSaved:")
print("../data/X_train.npy")
print("../data/X_val.npy")

print("\nDONE")"""

print("Stage 3 Complete.")