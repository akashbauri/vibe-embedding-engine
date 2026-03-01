import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def generate_embeddings(frame_paths):
    embeddings = []

    for path in frame_paths:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.get_image_features(**inputs)

        # 🔧 FIX: extract tensor properly
        emb = output.detach().cpu().numpy()[0]

        embeddings.append(emb)

    embeddings = np.array(embeddings)

    # Verification
    assert not np.isnan(embeddings).any(), "NaN detected"

    return embeddings
