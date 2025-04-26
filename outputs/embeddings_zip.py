import zipfile

with zipfile.ZipFile("log_embeddings.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
    zipf.write("outputs/log_embeddings.npy", arcname="log_embeddings.npy")