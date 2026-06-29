import pandas as pd

selected = pd.read_csv(
    "../config/selected_features.csv"
)

features = selected.iloc[:,0].tolist()

row = {}

for f in features:
    row[f] = telemetry.get(
        f,
        np.nan
    )

df = pd.DataFrame([row])

print(df.shape)
print(df.head())
print(
    df.isna().sum().sum()
)

"""Convert websocket telemetry
into the same format as
train_jun9_10_clean.csv
Expected:

(1,139)"""