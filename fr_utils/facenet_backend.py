# fr_utils/facenet_backend.py
# Robust FaceNet embedding helpers with single-image and batch APIs.

import cv2
import numpy as np
from pathlib import Path
from typing import Iterable, List, Tuple
from keras_facenet import FaceNet  # pip install keras-facenet

# Lazy singletons to avoid reloading between calls
_EMBEDDER = None

def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        # keras-facenet returns 512-D embeddings, expects 160x160 RGB
        _EMBEDDER = FaceNet()
    return _EMBEDDER

def _ensure_rgb_160(img_bgr: np.ndarray) -> np.ndarray:
    """Ensure an aligned face image is resized to 160x160 and converted to RGB."""
    if img_bgr is None or not hasattr(img_bgr, "shape"):
        raise ValueError("Invalid image (None or not a numpy array).")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (160, 160), interpolation=cv2.INTER_LINEAR)
    return img_rgb

def embed_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Embed a single aligned face (BGR array). Returns shape (512,).
    """
    embedder = _get_embedder()
    rgb = _ensure_rgb_160(img_bgr)
    vec = embedder.embeddings([rgb])[0]  # (512,)
    return vec.astype("float32")

def _read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    return img

def embed_faces(paths: Iterable[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Embed a list of image file paths (aligned faces expected).
    Returns (embeddings[N,512], kept_paths[List[str]]). Any unreadable files are skipped.
    """
    embedder = _get_embedder()
    kept_rgb: List[np.ndarray] = []
    kept_paths: List[str] = []

    for p in paths:
        img = _read_bgr(Path(p))
        if img is None:
            # skip unreadable
            continue
        try:
            rgb = _ensure_rgb_160(img)
        except Exception:
            continue
        kept_rgb.append(rgb)
        kept_paths.append(p)

    if not kept_rgb:
        # Return empty array with correct dims
        return np.zeros((0, 512), dtype="float32"), []

    embs = embedder.embeddings(kept_rgb).astype("float32")
    return embs, kept_paths
