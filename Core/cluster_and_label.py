import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# === Load Data ===
embeddings = np.load("outputs/log_embeddings.npy")
df = pd.read_csv("outputs/cleaned_logs.csv")

# === Clustering using DBSCAN ===
print("🔄 Running DBSCAN clustering...")
clustering = DBSCAN(eps=1.5, min_samples=5)
cluster_labels = clustering.fit_predict(embeddings)

df["cluster_id"] = cluster_labels

# === Show samples per cluster ===
print("\n🧩 Sample logs from each cluster:\n")
for cluster_id in sorted(set(cluster_labels)):
    if cluster_id == -1:
        continue  # skip noise
    samples = df[df["cluster_id"] == cluster_id].head(5)["log_message"].tolist()
    print(f"🔹 Cluster {cluster_id} ({len(samples)} logs):")
    for s in samples:
        print("   •", s[:120])
    print()

# === Auto-label each cluster using top TF-IDF words ===
print("🧠 Generating labels for each cluster...")
label_map = {}
tfidf = TfidfVectorizer(max_features=3, stop_words="english")

for cluster_id in set(cluster_labels):
    if cluster_id == -1:
        continue
    logs = df[df["cluster_id"] == cluster_id]["log_message_clean"].tolist()
    if len(logs) == 0:
        continue
    tfidf_matrix = tfidf.fit_transform(logs)
    top_keywords = tfidf.get_feature_names_out()
    label_map[cluster_id] = "_".join(top_keywords)

# === Map cluster_id to human-readable label ===
df["cluster_label"] = df["cluster_id"].apply(lambda x: label_map.get(x, "Noise"))

# === Save labeled data for training ===
final_df = df[["log_message", "cluster_label"]]
os.makedirs("outputs", exist_ok=True)
final_df.to_csv("outputs/labeled_logs.csv", index=False)

print("✅ Labeled logs saved to outputs/labeled_logs.csv")
