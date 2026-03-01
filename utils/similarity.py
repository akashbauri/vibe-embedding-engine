import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(embeddings):
    sim_matrix = cosine_similarity(embeddings)

    # Verification
    assert sim_matrix[0][0] > 0.99

    return sim_matrix
