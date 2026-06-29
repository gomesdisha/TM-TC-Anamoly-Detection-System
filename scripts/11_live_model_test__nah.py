import joblib
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