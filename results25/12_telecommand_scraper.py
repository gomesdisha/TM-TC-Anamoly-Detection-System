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
