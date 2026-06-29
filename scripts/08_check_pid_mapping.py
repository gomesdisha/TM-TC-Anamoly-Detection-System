import pandas as pd
import requests

PID_URL = "http://172.20.10.1:9000/pid_info?sc_id=EOS-10"

selected = pd.read_csv("../config/selected_features.csv")

print("Loading PID mapping...")

resp = requests.get(PID_URL, timeout=20)
data = resp.json()

pid_map = {}

for item in data:
    if "mnemonic" in item:
        pid_map[item["mnemonic"].upper()] = item["pid"]

print("Total spacecraft parameters:", len(pid_map))

features = selected.iloc[:,0].tolist()

matches = [
    f for f in features
    if f.upper() in pid_map
]

print("Selected Features:", len(features))
print("Matched Features :", len(matches))

print("\nFirst 20 Matches:")
for m in matches[:20]:
    print(m)

"""Can I see all spacecraft parameters?
How many parameters exist?
How many of my 139 features exist?"""