# fr_utils/open_set.py
from __future__ import annotations
import json, os
import numpy as np
from typing import Dict, Tuple, List
from sklearn.neighbors import KNeighborsClassifier

def compute_centroids(X: np.ndarray, y: np.ndarray, classes: List[str]) -> np.ndarray:
    """Mean embedding per class in 'classes' order."""
    centroids = []
    for c in classes:
        mask = (y == c)
        if not np.any(mask):
            # should not happen if enroll has all classes
            centroids.append(np.zeros(X.shape[1], dtype=np.float32))
        else:
            centroids.append(X[mask].mean(axis=0))
    return np.vstack(centroids).astype(np.float32)

def l2_dist(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    return float(np.linalg.norm(a - b))

def distances_to_centroid(embs: np.ndarray, centroids: np.ndarray, class_idx: np.ndarray) -> np.ndarray:
    """Per-sample distance to its own class centroid."""
    return np.linalg.norm(embs - centroids[class_idx], axis=1)

def calibrate_threshold(
    enroll_dists: np.ndarray,
    far_quantile: float = 0.95,
    headroom: float = 1.10,
) -> float:
    """
    Set a distance threshold so ~ (1 - far_quantile) of genuine enroll samples would be rejected.
    Add a small headroom >1 to be conservative.
    """
    base = np.quantile(enroll_dists, far_quantile)
    return float(base * headroom)

def save_open_set_cfg(
    path: str,
    prob_threshold: float,
    dist_threshold: float,
    note: str = "",
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "prob_threshold": prob_threshold,
                "dist_threshold": dist_threshold,
                "note": note,
            },
            f,
            indent=2,
        )

def load_open_set_cfg(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
