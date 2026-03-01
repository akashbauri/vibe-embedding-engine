import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os

from utils.extract import extract_frames
from utils.embed import generate_embeddings
from utils.similarity import compute_similarity

st.set_page_config(page_title="GACS Mini Prototype", layout="wide")

st.title("🎨 GACS Mini Prototype")
st.subheader("Mood & Style Similarity Engine for Art / Marketing Videos")

uploaded_video = st.file_uploader("Upload Art / Marketing Video", type=["mp4"])

if uploaded_video:

    try:
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.info("Step 1: Extracting frames...")
        frames = extract_frames(video_path)

        if len(frames) == 0:
            st.error("No frames extracted. Please upload a valid video.")
        else:
            st.success(f"{len(frames)} frames extracted")

            st.info("Step 2: Generating embeddings...")
            embeddings = generate_embeddings(frames)

            # 🔍 Verification Logs
            st.write("Embedding Shape:", embeddings.shape)

            if np.isnan(embeddings).any():
                st.error("NaN values detected in embeddings!")
            else:
                st.success("Embeddings verified: No NaN values")

            st.info("Step 3: Computing vibe similarity...")
            sim_matrix = compute_similarity(embeddings)

            st.success("Similarity matrix computed")

            st.subheader("📊 Similarity Heatmap")

            fig, ax = plt.subplots()
            cax = ax.imshow(sim_matrix)
            fig.colorbar(cax)
            st.pyplot(fig)

            # 🔎 Top-5 Similar Frames Retrieval
            st.subheader("🔎 Top-5 Similar Frames Retrieval")

            query_index = 0
            similarities = sim_matrix[query_index]

            top5 = similarities.argsort()[-6:-1][::-1]

            st.write(f"Query Frame Index: {query_index}")
            st.write("Top 5 Most Similar Frames:", top5.tolist())

            # 🔬 Self Similarity Check (Verification-first mindset)
            if sim_matrix[0][0] > 0.99:
                st.success("Self-similarity verification passed")
            else:
                st.warning("Self-similarity check failed")

            st.success("🎯 Vibe similarity analysis completed!")

    except Exception as e:
        st.error(f"Pipeline failed: {e}")
