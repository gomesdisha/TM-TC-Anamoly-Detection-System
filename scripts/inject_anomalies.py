import pandas as pd

df = pd.read_csv("../data/test_jun11_clean.csv")

# -------------------
# Temperature spike
# -------------------

df.loc[
    5000:5200,
    "SPS1_BASE_M_TS026_TMP"
] = 40

# -------------------
# Voltage spike
# -------------------

df.loc[
    15000:15200,
    "BUS_VOL_SEL_RT"
] = 45

# -------------------
# Battery anomaly
# -------------------

df.loc[
    25000:25500,
    "BAT_CUR_RAW_SEL_RT"
] = -50

df.to_csv(
    "../data/test_jun11_injected.csv",
    index=False
)

print("Done")

