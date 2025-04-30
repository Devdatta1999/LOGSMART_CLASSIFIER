import re
import csv
from pathlib import Path

# === File Paths ===
INPUT_FILE = "data/raw_logs/raw_macos_logs.log"
OUTPUT_FILE = "data/structured_logs/structured_macos_logs.csv"

# === Output folder creation ===
Path("data/structured_logs").mkdir(parents=True, exist_ok=True)

# === Regex to match a structured log line ===
log_pattern = re.compile(
    r'^(?P<date>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+(?P<source>\S+)\s+(?P<rest>.+)$'
)

structured_data = []

with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()

        # Skip empty or repeat lines
        if not line or line.startswith("--- last message repeated"):
            continue

        match = log_pattern.match(line)
        if match:
            log = match.groupdict()
            rest = log["rest"]

            # Split the rest on the FIRST colon to get the log message
            if ':' in rest:
                process_info, message = rest.split(':', 1)
                structured_data.append({
                    "date": log["date"],
                    "source": log["source"],
                    "log_message": message.strip()
                })

# === Write to CSV ===
with open(OUTPUT_FILE, "w", newline='', encoding="utf-8", errors="ignore") as f:
    writer = csv.DictWriter(f, fieldnames=["date", "source", "log_message"])
    writer.writeheader()
    writer.writerows(structured_data)

print(f"✅ Structured CSV created at: {OUTPUT_FILE}")
print(f"📦 Total lines parsed: {len(structured_data)}")
