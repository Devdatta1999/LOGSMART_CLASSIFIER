# app.py

import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import os
from sentence_transformers import SentenceTransformer

# === App config ===
st.set_page_config(page_title="LogSmart", page_icon="📑", layout="wide")

# === Load models ===
@st.cache_resource
def load_resources():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    classifier = joblib.load('Models/log_classifier_model.joblib')
    return embedder, classifier

embedder, classifier = load_resources()

# === Normalization function (Same as training) ===
def normalize_log(text):
    text = text.lower()
    text = re.sub(r'\d+', '<NUM>', text)  # replace numbers
    text = re.sub(r'http\S+', 'httpurl', text)  # replace URLs
    text = re.sub(r'\s+', ' ', text)  # normalize spaces
    return text.strip()

# === Sidebar ===
st.sidebar.title("🛠️ Options")
st.sidebar.markdown("Upload your MacOS raw logs to classify them automatically into categories.")

# === Main page ===
st.title("📑 LogSmart Classifier")
st.write("A professional tool to **classify MacOS logs** automatically based on pretrained ML model.")

uploaded_file = st.file_uploader("📂 Upload your raw log file (.csv)", type=["csv"])

if uploaded_file:
    try:
        with st.spinner('🔄 Processing... Please wait.'):
            df = pd.read_csv(uploaded_file)

            # Validate required columns
            required_cols = ['date', 'source', 'log_message']
            if not all(col in df.columns for col in required_cols):
                st.error("❌ Uploaded file must have columns: `date`, `source`, and `log_message`.")
            else:
                st.success("✅ File validated successfully!")

                # Normalize log_message
                df['log_message_clean'] = df['log_message'].astype(str).apply(normalize_log)

                # Embed
                embeddings = embedder.encode(df['log_message_clean'].tolist(), show_progress_bar=True)

                # Predict labels
                preds = classifier.predict(embeddings)

                # Final classified dataframe
                df['log_category'] = preds
                final_df = df[['date', 'source', 'log_message', 'log_category']]
                
                # --- Analytics Section ---
                import matplotlib.pyplot as plt

                

                # Show preview
                st.subheader("🔍 Preview of classified logs:")
                st.dataframe(final_df.head(10))

                


                # Download button
                st.download_button(
                    label="📥 Download Classified Logs",
                    data=final_df.to_csv(index=False).encode('utf-8'),
                    file_name='classified_logs.csv',
                    mime='text/csv'
                )

                st.markdown("## 📊 Log Category Distribution")

                # Group by log_category and count
                category_counts = final_df["log_category"].value_counts()

                # Plot
                fig, ax = plt.subplots(figsize=(10, 5))
                category_counts.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
                plt.title('Number of Logs per Category', fontsize=16)
                plt.xlabel('Log Category', fontsize=12)
                plt.ylabel('Number of Logs', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

                st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error occurred: {e}")

# Footer
st.markdown("---")
st.caption("Developed by LogSmart Team | Powered by Streamlit ❤️")
