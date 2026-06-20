import pandas as pd

print("Loading...")

df = pd.read_csv(
    "../data/all-pids-4days_clean.csv"
)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%d-%b-%Y %H:%M:%S.%f"
)

train_df = df[
    df["Timestamp"].dt.day.isin([9,10])
]

test_df = df[
    df["Timestamp"].dt.day == 11
]

print("Train:", train_df.shape)
print("Test :", test_df.shape)

train_df.to_csv(
    "../data/train_jun9_10.csv",
    index=False
)

test_df.to_csv(
    "../data/test_jun11.csv",
    index=False
)

print("Done")