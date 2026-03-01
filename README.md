# vibe-embedding-engine
Mini affective computing engine for measuring visual mood &amp; style similarity in art and marketing videos.


🚀 Project Objective

The goal of this prototype is to simulate a core component of GenTA’s future GACS (Generative Affective Computing System):

Converting visual aesthetics into embeddings that represent emotional tone.

This allows machines to interpret how visuals feel, not just what they contain.

🧠 System Pipeline

The application performs the following steps:

Video Ingestion

Upload short art or marketing-style videos via Streamlit

Frame Extraction

Representative frames extracted at time intervals

Multimodal Embedding

CLIP model converts frames into 512-dimensional vectors

Similarity Computation

Pairwise cosine similarity between frames

Visualization

Heatmap showing mood/style similarity

Top-5 Retrieval

Finds most emotionally similar frames

📊 Output Example

The system generates:

Frame embeddings

Similarity matrix

Heatmap visualization

Top-5 vibe similarity retrieval

Frames with similar lighting, motion, and composition cluster together — indicating shared emotional tone.

✅ Verification-First Engineering

To ensure robustness:

Embedding shape validation

NaN checks

Self-similarity assertion

Exception-safe execution

This reflects a research-oriented pipeline design.

🤖 AI Tool Usage

AI coding tools were used for:

Initial CLIP integration

Streamlit deployment

Similarity logic acceleration

All outputs were manually verified and debugged.

🔮 Future GACS Extension

This prototype can evolve into:

Automatic creative mood scoring

Audience-emotion alignment

Emotion-aware ad optimization

Next Steps

Map embeddings to CTR / ROAS

Build affect-performance feedback loop

🛠 Tech Stack

Python

PyTorch

OpenAI CLIP (via HuggingFace)

OpenCV

Streamlit

Scikit-learn

Matplotlib

▶️ Run Locally
pip install -r requirements.txt
streamlit run app.py
🌐 Live Demo

Deployed on Streamlit Cloud
Upload any art / marketing video to explore vibe similarity.

📌 Impact

This system demonstrates how machines can interpret visual mood — making contemporary art and marketing creatives more emotionally accessible and optimizable.
