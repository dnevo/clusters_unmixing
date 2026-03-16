
# plots.py
# Visualization utilities

import plotly.graph_objects as go
import numpy as np

def plot_cluster_overview(clusters: np.ndarray):
    fig = go.Figure()
    for i, s in enumerate(clusters):
        fig.add_trace(go.Scatter(y=s, mode="lines", name=f"cluster {i+1}"))
    fig.update_layout(title="Cluster spectra overview", xaxis_title="band", yaxis_title="reflectance")
    fig.show()
