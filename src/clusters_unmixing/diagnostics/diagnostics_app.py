
# diagnostics_app.py
# Notebook orchestration layer

from .diagnostics import cosine_similarity_matrix
from .plots import plot_cluster_overview
import numpy as np

def run_cluster_diagnostics():
    # simple demo orchestration
    clusters = np.random.rand(6, 100)
    sim = cosine_similarity_matrix(clusters)
    print("Cosine similarity matrix:")
    print(sim)
    plot_cluster_overview(clusters)
