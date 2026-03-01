import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os

from utils.extract import extract_frames
from utils.embed import generate_embeddings
from utils.similarity import compute_similarity

st.title("🎨 GACS Mini Prototype")
st.write("Mood & Style Similarity Engine")

uploaded_video = st.file_uploader("Upload Art / Marketing Video", type=["mp4"])

if uploaded_video:

    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.write("Extracting frames...")
    frames = extract_frames(video_path)

    st.write(f"{len(frames)} frames extracted")

    st.write("Generating embeddings...")
    embeddings = generate_embeddings(frames)

    st.write("Computing vibe similarity...")
    sim_matrix = compute_similarity(embeddings)

    st.write("### Similarity Heatmap")

    fig, ax = plt.subplots()
    cax = ax.imshow(sim_matrix)
    fig.colorbar(cax)
    st.pyplot(fig)

    st.success("Vibe similarity computed successfully!")
