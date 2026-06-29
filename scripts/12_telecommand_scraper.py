"""
12_telecommand_scraper.py
------------------------------------
Scrapes SEND telecommands from ISRO webpage.
"""

import os
import pandas as pd
import requests
from bs4 import BeautifulSoup

URL = "http://172.20.10.1:8888/PEPSUMMARY/Summary.jsp?ScName=EOS-10&STime=null&ETime=null"

def scrape():
    print(f"🌐 Fetching: {URL}")
    try:
        response = requests.get(URL, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Fetch failed: {e}")
        return

    soup = BeautifulSoup(response.text, "html.parser")
    new_commands = []

    # 🔹 The page has multiple procedure tables. Scan ALL of them.
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cols = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cols) < 4:
                continue

            # 🔹 Column mapping based on actual HTML structure:
            # cols[0] -> STEP ID
            # cols[1] -> INSTRUCTION (SEND, END, SET, etc.)
            # cols[2] -> INFO (Command details/text)
            # cols[3] -> TIME & TIMESTAMP

            instruction = cols[1].strip().upper()

            if instruction == "SEND":
                timestamp = cols[3].strip()
                command_detail = cols[2].strip()

                # Clean up multi-line or textarea content
                if "\n" in command_detail:
                    command_detail = command_detail.split("\n")[0].strip()

                new_commands.append({
                    "Timestamp": timestamp,
                    "Command": instruction,
                    "Details": command_detail
                })

    if not new_commands:
        print("⚠️ No 'SEND' commands found on page.")
        return

    df = pd.DataFrame(new_commands)

    # 🔹 Append mode to build command history without overwriting
    output_file = "live_commands.csv"
    file_exists = os.path.exists(output_file)

    df.to_csv(output_file, mode="a", header=not file_exists, index=False)

    print(f"✅ Found {len(df)} SEND command(s)")
    print(df.tail())
    print(f"📁 Saved to: {output_file}")

if __name__ == "__main__":
    scrape()


"""
=========================================================
12_telecommand_scraper.py

Continuously monitors telecommand webpage.
Extracts ONLY SEND commands.
Saves only NEW commands.

Output:
live_commands.csv

Columns:
Timestamp
Command
=========================================================


import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

# =====================================================
# CONFIG
# =====================================================

URL = "http://172.20.10.1:8888/PEPSUMMARY/Summary.jsp?ScName=EOS-10&STime=null&ETime=null"

OUTPUT_FILE = "live_commands.csv"

CHECK_INTERVAL = 2      # seconds

# =====================================================
# LOAD PREVIOUS COMMANDS
# =====================================================

seen = set()

if os.path.exists(OUTPUT_FILE):

    old = pd.read_csv(OUTPUT_FILE)

    for _, row in old.iterrows():

        seen.add(
            (
                str(row["Timestamp"]),
                str(row["Command"])
            )
        )

print("Previously saved commands:", len(seen))

# =====================================================
# SCRAPER
# =====================================================

def get_commands():

    html = requests.get(
        URL,
        timeout=10
    ).text

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    commands = []

    tables = soup.find_all("table")

    for table in tables:

        rows = table.find_all("tr")

        for row in rows:

            cols = row.find_all("td")

            if len(cols) < 4:
                continue

            action = cols[1].get_text(
                strip=True
            ).upper()

            if action != "SEND":
                continue

            timestamp = cols[3].get_text(
                strip=True
            )

            textarea = cols[2].find(
                "textarea"
            )

            if textarea:

                lines = textarea.get_text().splitlines()

                for cmd in lines:

                    cmd = cmd.strip()

                    if cmd == "":
                        continue

                    commands.append(
                        (
                            timestamp,
                            cmd
                        )
                    )

            else:

                cmd = cols[2].get_text(
                    " ",
                    strip=True
                )

                commands.append(
                    (
                        timestamp,
                        cmd
                    )
                )

    return commands


# =====================================================
# MAIN LOOP
# =====================================================

print("\nMonitoring telecommands...\n")

while True:

    try:

        commands = get_commands()

        new_rows = []

        for timestamp, command in commands:

            key = (
                timestamp,
                command
            )

            if key in seen:
                continue

            seen.add(key)

            new_rows.append({

                "Timestamp": timestamp,

                "Command": command

            })

            print("=" * 60)
            print("NEW COMMAND")
            print("Timestamp :", timestamp)
            print("Command   :", command)
            print("=" * 60)

        if len(new_rows):

            df = pd.DataFrame(new_rows)

            if os.path.exists(OUTPUT_FILE):

                df.to_csv(

                    OUTPUT_FILE,

                    mode="a",

                    header=False,

                    index=False

                )

            else:

                df.to_csv(

                    OUTPUT_FILE,

                    index=False

                )

            print(
                f"Saved {len(new_rows)} new command(s)"
            )

        else:

            print("No new commands.")

    except Exception as e:

        print("ERROR :", e)

    time.sleep(CHECK_INTERVAL)
"""