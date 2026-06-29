import asyncio
import json
import requests
import numpy as np
import torch
import websockets

from bs4 import BeautifulSoup
from tcn_gru_autoencoder import TCN_GRU_Autoencoder

# ==========================================
# CONFIG
# ==========================================

WS_URL = "ws://172.20.10.1:9000/ws/ntm"
COMMAND_PAGE_URL = "http://172.20.10.1:8888/PEPSUMMARY/Summary.jsp?ScName=EOS-10"

WINDOW = 30
THRESHOLD = 0.0001

buffer = []

# ==========================================
# MODEL
# ==========================================

model = TCN_GRU_Autoencoder()

model.load_state_dict(
    torch.load(
        "tcn_gru_model_10f.pth",
        map_location="cpu"
    )
)

model.eval()


# ==========================================
# TELECOMMAND SCRAPING
# ==========================================

def get_latest_command():
    try:
        html = requests.get(COMMAND_PAGE_URL, timeout=5).text
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ")

        if "TCR_R_ON" in text:
            return "TCR_R_ON"
        elif "TCR_VOL&CUR_SET_R" in text:
            return "TCR_VOL&CUR_SET_R"
        return "NONE"
    except Exception:
        return "NONE"


# ==========================================
# COMMAND ENCODING
# ==========================================

def encode_command(cmd):
    mapping = {
        "NONE": 0,
        "TCR_R_ON": 1,
        "TCR_VOL&CUR_SET_R": 2
    }
    return mapping.get(cmd, 0)


# ==========================================
# TELEMETRY ENCODING
# ==========================================

def encode_value(value):
    if isinstance(value, str):
        value = value.upper()
        if value in ["ON", "LOCK", "TRUE", "POS-1"]:
            return 1.0
        if value in ["OFF", "UNLOCK", "FALSE", "POS-2"]:
            return 0.0
        return 0.0
    try:
        return float(value)
    except:
        return 0.0


# ==========================================
# FEATURE VECTOR
# ==========================================

def build_feature_vector(telemetry, command):
    keys = [
        "XSW-02_SW_POS", "XPA-M_DCDC-1_STS", "XPA-M_DCDC-2_STS",
        "XPA-R_DCDC-1_STS", "XPA-R_DCDC-2_STS", "PLD_TX-1_STS",
        "PLD_TX-2_STS", "PLD_TX-1_LOCK_STS", "PLD_TX-2_LOCK_STS"
    ]
    return [
        encode_value(telemetry.get(k, 0))
        for k in keys
    ] + [encode_command(command)]


# ==========================================
# DETECTION
# ==========================================

def detect(window):
    x = torch.tensor(np.array(window, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        reconstructed = model(x)
        error = torch.mean((x - reconstructed) ** 2).item()

    status = "NORMAL" if error <= THRESHOLD else "ANOMALY"
    print(f"Score={error:.6f} | Threshold={THRESHOLD:.6f} | Status={status}")
    return status


# ==========================================
# PID MAPPING & WEBSOCKET CLIENT
# ==========================================

def fetch_pid_mapping():
    url = "http://172.20.10.1:9000/pid_info?sc_id=EOS-10"
    print(f"🔍 Fetching PID map from: {url}")
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, list):
            print(f" Unexpected structure: {type(data)}")
            return {}, {}

        # Go struct uses lowercase keys: "pid", "mnemonic"
        pid_map = {}  # mnemonic -> pid
        reverse_pid_map = {}  # pid -> mnemonic

        for item in data:
            if isinstance(item, dict) and "mnemonic" in item and "pid" in item:
                mnemonic = item["mnemonic"].upper()
                pid = item["pid"]
                pid_map[mnemonic] = pid
                reverse_pid_map[pid] = mnemonic

        print(f"Loaded {len(pid_map)} PID mappings")
        return pid_map, reverse_pid_map
    except Exception as e:
        print(f" PID fetch failed: {e}")
        return {}, {}


# Initialize PID maps
PID_MAP, REVERSE_PID_MAP = fetch_pid_mapping()

TARGET_MNEMONICS = [
    "XSW-02_SW_POS", "XPA-M_DCDC-1_STS", "XPA-M_DCDC-2_STS",
    "XPA-R_DCDC-1_STS", "XPA-R_DCDC-2_STS", "PLD_TX-1_STS",
    "PLD_TX-2_STS", "PLD_TX-1_LOCK_STS", "PLD_TX-2_LOCK_STS"
]

# Extract PIDs for target mnemonics
TARGET_PIDS = [PID_MAP[m] for m in TARGET_MNEMONICS if m in PID_MAP]
print(f" Subscribing to PIDs: {TARGET_PIDS}")


async def main():
    if not TARGET_PIDS:
        print(" No valid PIDs found. Check PID mapping.")
        return

    try:
        async with websockets.connect(
                WS_URL,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5
        ) as ws:
            print(" Connected to WS")

            # Construct subscription request matching Go 'request' struct
            sub_payload = {
                "user_id": "PRISM",
                "msg_type": "ntm",
                "msg_payload": {
                    "sc_id": "EOS-10",
                    "stream": "ANY-TM1",
                    "action": "subscribe",
                    "parameters": TARGET_PIDS
                },
                "on_change": False
            }

            print(" Sending subscription request...")
            await ws.send(json.dumps(sub_payload))
            print(" Subscription sent")

            print("\n Listening for telemetry...")
            msg_count = 0
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=300)
                    msg_count += 1
                    print(f" [{msg_count}] {msg[:300]}...")

                    # Parse response matching Go 'response' and 'responsePayload' structs
                    telemetry = {}
                    try:
                        resp = json.loads(msg)
                        payload = resp.get("msg_payload", {})
                        params_info = payload.get("parameters_info", [])

                        for p in params_info:
                            pid = p.get("param")
                            mnemonic = REVERSE_PID_MAP.get(pid)
                            if mnemonic:
                                # Value extraction matching Go 'Parameter' struct
                                val = p.get("str_v") if p.get("str_v") else p.get("float_v")
                                telemetry[mnemonic] = val
                                print(f"   {mnemonic} ({pid}): {val}")
                    except json.JSONDecodeError as je:
                        print(f" Invalid JSON: {je}")
                    except Exception as proc_err:
                        print(f" Processing error: {proc_err}")

                    # Build feature vector and detect
                    command = await asyncio.to_thread(get_latest_command)
                    fv = build_feature_vector(telemetry, command)
                    buffer.append(fv)
                    if len(buffer) > WINDOW:
                        buffer.pop(0)
                    if len(buffer) == WINDOW:
                        detect(buffer)

                except asyncio.TimeoutError:
                    print(" Timeout waiting for telemetry")
                    break
                except websockets.exceptions.ConnectionClosed as e:
                    print(f" Connection closed: {e}")
                    break
                except Exception as e:
                    print(f" Unexpected error: {e}")
                    break

    except Exception as e:
        print(f" Connection failed: {e}")


if __name__ == "__main__":
    print("Starting WebSocket telemetry listener...")
    asyncio.run(main())

