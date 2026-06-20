"""import pandas as pd

features = pd.read_csv(
    "config/selected_features.csv"
).iloc[:,0].tolist()

print("Feature Count:", len(features))
print(features[:10])"""

import pandas as pd

features = pd.read_csv("config/selected_features.csv")

print(features.shape)

print(features.tail(10))

"""features = pd.read_csv("config/selected_features.csv")

print(features.duplicated().sum())"""
