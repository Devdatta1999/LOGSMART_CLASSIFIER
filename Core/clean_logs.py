import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import os

def normalize_log(text):
    text = text.lower()
    text = re.sub(r'\d+', '<NUM>', text)  # replace numbers
    text = re.sub(r'http\S+', 'httpurl', text)  # replace URLs
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# === Load structured log file ===
df = pd.read_csv("data/structured_logs/structured_macos_logs.csv")

# === Normalize log_message column ===
df["log_message_clean"] = df["log_message"].astype(str).apply(normalize_log)


# Save cleaned logs for reuse
os.makedirs("outputs", exist_ok=True)

df[["log_message", "log_message_clean"]].to_csv("outputs/cleaned_logs.csv", index=False)

print("✅logs saved.")