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


# === Load SentenceTransformer model ===
print("🔄 Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === Generate embeddings for log_message_clean ===
print("🔢 Generating embeddings...")
embeddings = embedder.encode(df["log_message_clean"].tolist(), show_progress_bar=True)

# Just show shape of the first vector
print("✅ Sample embedding vector length:", len(embeddings[0]))

# Save embeddings and cleaned logs for reuse
os.makedirs("outputs", exist_ok=True)
np.save("outputs/log_embeddings.npy", embeddings)
df[["log_message", "log_message_clean"]].to_csv("outputs/cleaned_logs.csv", index=False)

print("✅ Embeddings and logs saved.")