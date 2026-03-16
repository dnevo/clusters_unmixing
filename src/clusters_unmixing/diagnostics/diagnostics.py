
# diagnostics.py
# Analysis logic only (no plotting)

import numpy as np
import pandas as pd

def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    return Xn @ Xn.T

def spectral_angle_mapper(a: np.ndarray, b: np.ndarray) -> float:
    num = np.dot(a, b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.arccos(np.clip(num / den, -1.0, 1.0)))

def abundance_error_table(df: pd.DataFrame) -> pd.DataFrame:
    true_cols = [c for c in df.columns if c.startswith("true_a")]
    pred_cols = [c for c in df.columns if c.startswith("est_a")]
    rows = []
    for _, r in df.iterrows():
        t = r[true_cols].to_numpy(dtype=float)
        p = r[pred_cols].to_numpy(dtype=float)
        rows.append({
            "pixel_index": int(r.get("pixel_index", -1)),
            "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
            "mae": float(np.mean(np.abs(p - t)))
        })
    return pd.DataFrame(rows)
