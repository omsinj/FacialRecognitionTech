"""
prototype_access.py (or quick single-image open-set decision script)

Purpose
-------
Given a single aligned face image, this script:

    1. Loads the trained closed-set classifier and label encoder.
    2. Loads per-class centroids and intra-class radii from centroids.npz.
    3. Embeds the input image using the FaceNet backend.
    4. Obtains classifier probabilities for all enrolled identities.
    5. Applies an open-set gate that combines:
           - classifier confidence (class probability), and
           - distance to the predicted-class centroid,
       to decide whether access is GRANTED or DENIED (UNKNOWN).

This replicates the decision rule used by the prototype access-control artefact
and is useful for manual testing from the command line.

Inputs
------
    • A single face image path passed via --image.
      The image is expected to contain a reasonably centred, frontal face.
      (Any required detection/alignment should already be handled in the
       training pipeline or by the embed_image function.)

Dependencies
-----------
    • models/classifier.joblib      – k-NN or SVM classifier
    • models/label_encoder.joblib   – LabelEncoder mapping identity strings → ints
    • data/meta/centroids.npz       – labels, centroids, radii computed offline
"""

import argparse
import os

import cv2
import numpy as np
from joblib import load

from fr_utils.config import MODELS_DIR, META_DIR
from fr_utils.facenet_backend import embed_image

# Path to centroid artefacts produced by scripts/build_centroids.py or similar
CENTROIDS_PATH = os.path.join(META_DIR, "centroids.npz")


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """
    L2-normalise a vector.

    Parameters
    ----------
    v : np.ndarray
        Input vector (any shape with last dimension as features).

    Returns
    -------
    np.ndarray
        L2-normalised vector. A small epsilon is added to avoid division by zero.
    """
    return v / (np.linalg.norm(v) + 1e-8)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine distance between two vectors.

    Distance is defined as:

        d_cos(a, b) = 1 - cos(a, b)
                    = 1 - (a · b / (||a|| * ||b||))

    After L2 normalisation, this reduces to 1 - (a · b).

    Parameters
    ----------
    a, b : np.ndarray
        Input vectors.

    Returns
    -------
    float
        Cosine distance in [0, 2]; smaller values indicate higher similarity.
    """
    a = l2_normalize(a)
    b = l2_normalize(b)
    return 1.0 - float(np.dot(a, b))  # smaller is closer


def decide_open_set(
    prob_vec: np.ndarray,
    class_idx: int,
    emb: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    radii: np.ndarray,
    prob_thr: float = 0.80,
    radius_mult: float = 1.2,
):
    """
    Open-set acceptance rule combining probability and centroid distance.

    A prediction is ACCEPTED if BOTH of the following hold:
        1. The classifier's probability for the predicted class
           is at least prob_thr.
        2. The cosine distance between the embedding and that class's centroid
           is less than or equal to radius_mult * radius[class].

    Otherwise, the sample is rejected as UNKNOWN.

    This encodes a simple per-class adaptive decision boundary:

        • prob_thr   – global minimum classifier confidence.
        • radius     – mean intra-class distance around each centroid.
        • radius_mult – scalar > 1 that slightly expands each class region.

    Parameters
    ----------
    prob_vec : np.ndarray
        Vector of class probabilities for a single sample (shape: [C]).
    class_idx : int
        Index of the predicted class in prob_vec and in centroids/radii arrays.
    emb : np.ndarray
        Embedding vector for the probe sample (1D).
    labels : np.ndarray
        Array of label strings aligned with centroids/radii (unused here but
        kept for clarity and debugging).
    centroids : np.ndarray
        Centroid matrix of shape (C, D), one centroid per class.
    radii : np.ndarray
        Mean intra-class radius per class (length C).
    prob_thr : float, optional
        Global minimum class probability for acceptance, by default 0.80.
    radius_mult : float, optional
        Multiplier applied to class radius for distance gating, by default 1.2.

    Returns
    -------
    tuple[bool, float, float, float]
        (accept, top_prob, distance, distance_threshold)
            accept              – True if sample passes the open-set gate.
            top_prob            – classifier probability for predicted class.
            distance            – cosine distance to predicted-class centroid.
            distance_threshold  – effective distance gate for that class.
    """
    top_prob = float(prob_vec[class_idx])

    # distance to predicted class centroid (in cosine geometry)
    c = centroids[class_idx]
    d = cosine_distance(emb, c)

    # per-class adaptive threshold: radius scaled by multiplier
    r = float(radii[class_idx]) * radius_mult

    accept = (top_prob >= prob_thr) and (d <= r)
    return accept, top_prob, d, r


def main(image_path: str, prob_thr: float, radius_mult: float):
    """
    Command-line entry point.

    Steps
    -----
    1. Load classifier and label encoder from MODELS_DIR.
    2. Load centroids and intra-class radii from CENTROIDS_PATH.
    3. Read the supplied image path and embed via FaceNet.
    4. Compute class probabilities with the classifier.
    5. Align classifier prediction with centroid/radius indices.
    6. Apply the open-set decision rule (decide_open_set).
    7. Print a human-readable decision line to stdout.

    Parameters
    ----------
    image_path : str
        Path to the face image to be evaluated.
    prob_thr : float
        Minimum classifier probability for acceptance.
    radius_mult : float
        Multiplicative factor applied to per-class radius in distance gating.
    """
    # --- Load models ---
    clf = load(os.path.join(MODELS_DIR, "classifier.joblib"))
    le  = load(os.path.join(MODELS_DIR, "label_encoder.joblib"))

    # --- Load centroids ---
    # Saved with structured dtypes (labels: str, centroids: float32, radii: float32).
    data = np.load(CENTROIDS_PATH)  # saved with uniform dtypes -> no pickle needed
    labels = data["labels"]         # strings aligned to LabelEncoder classes below
    centroids = data["centroids"]   # shape (K, 512)
    radii = data["radii"]           # shape (K,)

    # --- Read & embed image ---
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise SystemExit(f"Image not found: {image_path}")
    # embed_image encapsulates any required preprocessing (resize, normalisation)
    emb = embed_image(bgr).astype(np.float32).ravel()

    # --- Closed-set classification ---
    probs = clf.predict_proba(emb.reshape(1, -1))[0]
    pred_idx = int(np.argmax(probs))                   # index in label-encoder space
    pred_label = le.inverse_transform([pred_idx])[0]   # original identity label (string)

    # Align centroid index with label encoder order:
    #   - labels[] come from the centroid file
    #   - le.classes_ come from the LabelEncoder
    # We build a mapping from label string → row index in centroids/radii
    order = {lab: i for i, lab in enumerate(labels)}
    c_idx = order[str(pred_label)]

    # --- Open-set gate (probability + centroid distance) ---
    accept, top_prob, dist, dist_thr = decide_open_set(
        probs,
        c_idx,
        emb,
        labels,
        centroids,
        radii,
        prob_thr=prob_thr,
        radius_mult=radius_mult,
    )

    decision = "GRANTED" if accept else "DENIED (UNKNOWN)"
    print(
        f"Decision: {decision} | Pred: {pred_label} | "
        f"Prob: {top_prob:.3f} | Dist: {dist:.3f} | "
        f"thr(prob)≥{prob_thr:.2f} & dist≤{dist_thr:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-image open-set access decision using FaceNet embeddings."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the face image to evaluate (BGR-readable by OpenCV).",
    )
    parser.add_argument(
        "--prob_thr",
        type=float,
        default=0.80,
        help="Minimum classifier probability required to accept an identity.",
    )
    parser.add_argument(
        "--radius_mult",
        type=float,
        default=1.2,
        help="Multiplier applied to per-class centroid radius for distance gating.",
    )
    args = parser.parse_args()
    main(args.image, args.prob_thr, args.radius_mult)
