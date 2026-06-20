"""import pandas as pd

df = pd.read_csv("../data/all-pids_clean.csv")

print(df.shape)

print(df.isna().sum().sum())"""
"""import pandas as pd

df = pd.read_csv(
    "../data/all-pids-9-10Jun.csv",
    usecols=["Timestamp"],
    low_memory=False
)

print(df.head())
print(df.tail())
print(len(df))"""

#get size of file
"""import os

print(
    os.path.getsize(
        "../data/all-pids-9-10Jun.csv"
    ) / 1024 / 1024
)"""

# inspect_file.py
"""file = "../data/all-pids-9-10Jun.csv"

with open(file, "r", errors="ignore") as f:

    for i in range(5):
        line = f.readline()
        print(f"LINE {i}:")
        print(line[:300])
        print()"""

# check_csv_structure.py
"""file = "../data/all-pids-9-10Jun.csv"
with open(file, "r", errors="ignore") as f:

    for i in range(20):

        line = f.readline()

        print(
            i,
            len(line.split(","))
        )"""

"""import pandas as pd

features = pd.read_csv(
    "../config/selected_features.csv"
).iloc[:,0].tolist()

use_cols = ["Timestamp"] + features

chunks = []

for chunk in pd.read_csv(
        "../data/all-pids-9-10Jun.csv",
        usecols=use_cols,
        chunksize=5000):

    print(chunk.shape)

    chunks.append(chunk)

df = pd.concat(chunks)

print(df.shape)"""

#100 row tester 
"""import pandas as pd

features = pd.read_csv(
    "../config/selected_features.csv"
).iloc[:,0].tolist()

use_cols = ["Timestamp"] + features

df = pd.read_csv(
    "../data/all-pids-9-10Jun.csv",
    usecols=use_cols,
    nrows=100
)

print(df.shape)
print(df.head())"""

"""import pandas as pd

features = pd.read_csv(
    "../config/selected_features.csv"
).iloc[:,0].tolist()

use_cols = ["Timestamp"] + features

df = pd.read_csv(
    "../data/all-pids-9-10Jun.csv",
    usecols=use_cols,
    low_memory=False,
    memory_map=True
)

print(df.shape)"""

"""import pandas as pd

header = pd.read_csv(
    "../data/all-pids-9-10Jun.csv",
    nrows=0
)

print(len(header.columns))"""
#ram test

"""import psutil

print("RAM GB:",
      round(psutil.virtual_memory().total/1024**3,2))
print(psutil.virtual_memory())"""

"""import pandas as pd

features = pd.read_csv(
    "../config/selected_features.csv"
).iloc[:,0].tolist()

use_cols = ["Timestamp"] + features

total_rows = 0

for chunk in pd.read_csv(
        "../data/all-pids-9-10Jun.csv",
        usecols=use_cols,
        chunksize=1000,
        low_memory=False):

    print(chunk.shape)

    total_rows += len(chunk)

print("TOTAL:", total_rows)"""

"""import pandas as pd

df = pd.read_csv(
    "../data/all-pids-9-10Jun_clean.csv"
)

print(df.shape)
print("Total NaNs:", df.isna().sum().sum())"""

"""import pandas as pd

df = pd.read_csv(
    "../data/all-pids-9-10Jun_clean.csv",
    usecols=["Timestamp"]
)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%d-%b-%Y %H:%M:%S.%f"
)

delta = (
    df["Timestamp"]
    .diff()
    .dt.total_seconds()
)

print(delta.describe())"""
#check gpu-no gpu; 8gb ram
"""import torch

print(torch.cuda.is_available())"""

"""import pandas as pd

df = pd.read_csv(
    "../data/all-pids_clean.csv"
)

print(df.shape)
print(df.isna().sum().sum())"""

"""import json

metadata = {
    "seq_len": 60,
    "input_size": 139,
    "hidden_size": 32,
    "epochs": 10
}

with open(
    "../models/model_info.json",
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )"""
"""import pandas as pd
results = pd.read_csv(
    "../reports/june10_anomalies.csv"
)

print(
    results["Error"].describe()
)"""

"""import pandas as pd

df = pd.read_csv("../data/all-pids_clean.csv")

print(df.describe().T[["min","max"]].sort_values("max", ascending=False).head(20))"""
"""import pandas as pd

df = pd.read_csv("../data/all-pids_clean.csv")

for col in df.columns[1:]:
    if df[col].abs().max() > 1000:
        print(col, df[col].min(), df[col].max())"""
"""train = pd.read_csv("../data/all-pids-9-10Jun_clean.csv")
test = pd.read_csv("../data/all-pids_clean.csv")

print(train.shape)
print(test.shape)

print(
    list(train.columns) ==
    list(test.columns)
)"""



"""df = pd.read_csv("../reports/june10_anomalies.csv")

anoms = df[df["Anomaly"] == True]

print("Anomalies:", len(anoms))

print("\nFirst 20:")
print(anoms.head(20))

print("\nLast 20:")
print(anoms.tail(20))"""

"""import pandas as pd

df = pd.read_csv("../reports/june10_anomalies.csv")

anoms = df[df["Anomaly"] == True]

times = pd.to_datetime(
    anoms["Timestamp"],
    format="%d-%b-%Y %H:%M:%S.%f"
)

print(times.min())
print(times.max())"""

"""import pandas as pd

train = pd.read_csv("../data/all-pids-9-10Jun_clean.csv")
test = pd.read_csv("../data/all-pids_clean.csv")

train = train.drop(columns=["Timestamp"])
test = test.drop(columns=["Timestamp"])

diff = abs(
    train.mean() -
    test.mean()
)

print(
    diff.sort_values(
        ascending=False
    ).head(30)
)"""
"""import pandas as pd
df = pd.read_csv("../data/all-pids-4days_clean.csv")

print(df["Timestamp"].min())
print(df["Timestamp"].max())"""

"""import pandas as pd

df = pd.read_csv(
    "../data/all-pids-4days_clean.csv",
    usecols=["Timestamp"]
)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%d-%b-%Y %H:%M:%S.%f"
)

print(
    df["Timestamp"]
      .dt.date
      .value_counts()
      .sort_index()
)"""

"""import pandas as pd

df = pd.read_csv("../data/train_jun9_10.csv")

print(df.shape)
print(df.isna().sum().sum())"""

"""import pandas as pd

df = pd.read_csv("../data/train_jun9_10.csv")

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"]
)

split_idx = int(len(df) * 0.8)

print("START:")
print(df.iloc[0]["Timestamp"])

print("\nTRAIN END:")
print(df.iloc[split_idx-1]["Timestamp"])

print("\nVAL START:")
print(df.iloc[split_idx]["Timestamp"])

print("\nEND:")
print(df.iloc[-1]["Timestamp"])"""

"""import pandas as pd

df = pd.read_csv("../data/train_jun9_10.csv")

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"]
)

split_idx = int(len(df) * 0.8)

print(df.iloc[split_idx-5:split_idx+5][["Timestamp"]])"""

"""import pandas as pd

df = pd.read_csv("../data/train_jun9_10.csv")

split_idx = int(len(df) * 0.8)

train = df.iloc[:split_idx]
val = df.iloc[split_idx:]

for col in [
    "BUS_VOL_SEL_RT",
    "BUS_VOL_NSEL_RT",
    "BAT_CUR_RAW_SEL_RT",
    "OBC-1_UBUS_VOL",
    "OBC-2_UBUS_VOL"
]:
    print("\n", col)

    print(
        "Train:",
        train[col].mean(),
        train[col].std()
    )

    print(
        "Val:",
        val[col].mean(),
        val[col].std()
    )"""

"""import pandas as pd

df = pd.read_csv("../data/train_jun9_10.csv")

split_idx = int(len(df) * 0.8)

train = df.iloc[:split_idx]
val = df.iloc[split_idx:]

feature_drift = []

for col in df.columns:

    if col == "Timestamp":
        continue

    train_mean = train[col].mean()
    val_mean = val[col].mean()

    diff = abs(train_mean - val_mean)

    feature_drift.append(
        (col, diff)
    )

feature_drift.sort(
    key=lambda x: x[1],
    reverse=True
)

for x in feature_drift[:20]:
    print(x)"""

"""import pandas as pd

df = pd.read_csv("../data/train_jun9_10.csv")

split_idx = int(len(df) * 0.8)

train = df.iloc[:split_idx]
val = df.iloc[split_idx:]

print(train["BAT_CUR_RAW_SEL_RT"].describe())
print()
print(val["BAT_CUR_RAW_SEL_RT"].describe())"""

"""import pandas as pd

df = pd.read_csv("../data/train_jun9_10_clean.csv")

bad = df[df["BAT_CUR_RAW_SEL_RT"] < 0]

print("Count:", len(bad))

print()
print(bad[["Timestamp","BAT_CUR_RAW_SEL_RT"]].head(20))

print()
print(bad[["Timestamp","BAT_CUR_RAW_SEL_RT"]].tail(20))
"""
"""import pandas as pd

df = pd.read_csv("../data/train_jun9_10.csv")

print(
    df["BAT_CUR_RAW_SEL_RT"]
    .value_counts()
    .sort_index()
    .head(20)
)"""

"""import pandas as pd

df = pd.read_csv("../data/train_jun9_10.csv")

for col in [
    "BAT_RAW_CUR-M",
    "BAT_RAW_CUR-R",
    "BAT_CUR_RAW_SEL_RT"
]:
    print("\n", col)

    print(
        df[col]
        .value_counts()
        .head(10)
    )"""

"""import pandas as pd

df = pd.read_csv("../data/train_jun9_10.csv")

mask = df["BAT_CUR_RAW_SEL_RT"] == -26.59

cols = [
    "Timestamp",
    "BAT_RAW_CUR-M",
    "BAT_RAW_CUR-R",
    "BAT_CUR_RAW_SEL_RT"
]

print(df.loc[mask, cols].head(20))"""

"""import pandas as pd

df = pd.read_csv("../data/train_jun9_10.csv")

mask = df["BAT_CUR_RAW_SEL_RT"] == -26.59

print(mask.sum())

print(
    df.loc[mask, [
        "BAT_RAW_CUR-M",
        "BAT_RAW_CUR-R"
    ]].describe()
)"""

"""BAT_CUR_RAW_SEL_RT contained a constant value of -26.59 for
~2400 consecutive samples (10:29–10:55 on June 10).

The underlying battery current channels
BAT_RAW_CUR-M and BAT_RAW_CUR-R remained nominal.

Therefore BAT_CUR_RAW_SEL_RT was treated as an invalid/
derived telemetry artifact during preprocessing."""

"""import pandas as pd

df = pd.read_csv("../data/train_jun9_10_clean.csv")

print(
    (df["BAT_CUR_RAW_SEL_RT"] == -26.59).sum()
)"""

