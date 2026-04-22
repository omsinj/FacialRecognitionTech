"""
quick_accept_rate.py

Purpose
-------
Ad-hoc smoke-test for approximate 'acceptance rate' under a joint
(open-set inspired) probability + centroid-distance decision rule.

This script:
    • samples N test images from embeddings.parquet
    • re-embeds each aligned image using the online FaceNet inference path
    • obtains classifier probabilities (KNN)
    • checks acceptance via:
          - predicted-class probability >= prob_thr  (default 0.80)
          - cosine distance to class centroid <= radius * radius_mult
    • reports overall proportion accepted

Why this is useful
------------------
This simulates a simplified online access-decision using real embeddings
and model artefacts, providing a fast diagnostic of how strict thresholding
affects usability (accept rate). It is NOT an evaluation metric—it is a
debugging and exploratory tool prior to full open-set experiments.
"""

import os, random
import numpy as np
import pandas as pd
from joblib import load
import cv2

from fr_utils.config import META_DIR, MODELS_DIR, ALIGNED_DIR
from fr_utils.facenet_backend import embed_image


def main(n: int = 100,
         prob_thr: float = 0.80,
         radius_mult: float = 1.2) -> None:
    """
    Parameters
    ----------
    n : int
        Number of random test rows to sample (if fewer exist, uses all).
    prob_thr : float
        Minimum classifier softmax probability for acceptance.
    radius_mult : float
        Multiplier on the stored per-class centroid radius; a crude
        way to relax or tighten the distance tolerance.

    Behaviour
    ---------
    Loads test rows from embeddings.parquet and,
    for each:
        1. load aligned image from disk
        2. embed using embed_image (online embedding path)
        3. evaluate classifier probability
        4. compute cosine distance to predicted centroid
        5. check both decision gates and count acceptance
    """

    # --- Load embeddings metadata (for sampling only)
    df = pd.read_parquet(os.path.join(META_DIR, "embeddings.parquet"))
    feat_cols = [c for c in df.columns if str(c).isdigit()]  # numeric FaceNet dims
    test_rows = (
        df[df["split"] == "test"][["image_id", "identity"]]
        .sample(n=min(n, len(df)), random_state=42)
    )

    # --- Load model and label encoder
    clf = load(os.path.join(MODELS_DIR, "classifier.joblib"))
    le  = load(os.path.join(MODELS_DIR, "label_encoder.joblib"))

    # --- Load centroid store computed by build_centroids.py
    data = np.load(os.path.join(META_DIR, "centroids.npz"))
    labels    = data["labels"]
    centroids = data["centroids"]
    radii     = data["radii"]

    # label -> centroid index mapping
    order = {lab: i for i, lab in enumerate(labels)}

    granted = 0

    # Iterate over a sample of test images (not offline embeddings)
    for _, r in test_rows.iterrows():
        p = os.path.join(ALIGNED_DIR, r["image_id"])
        bgr = cv2.imread(p)
        if bgr is None:
            # missing file or unreadable; skip gracefully
            continue

        # embed the aligned image using the same online path used by the app
        emb = embed_image(bgr).astype(np.float32).ravel()

        # classifier probability over known classes
        probs = clf.predict_proba(emb.reshape(1, -1))[0]
        pred_idx = int(np.argmax(probs))
        pred_label = le.inverse_transform([pred_idx])[0]
        c_idx = order[str(pred_label)]

        # --- Cosine distance to the predicted centroid
        def l2n(v): return v / (np.linalg.norm(v) + 1e-8)
        dist = 1.0 - float(np.dot(l2n(emb), l2n(centroids[c_idx])))

        # --- Combine probability and centroid “radius” decision
        accept = (
            float(probs[c_idx]) >= prob_thr
            and dist <= float(radii[c_idx]) * radius_mult
        )
        granted += int(accept)

    # Simple acceptance statistic
    rate = granted / max(1, len(test_rows))
    print(
        f"Accepted {granted}/{len(test_rows)} | "
        f"Acceptance rate: {rate:.3f} "
        f"(prob_thr={prob_thr}, radius_mult={radius_mult})"
    )


if __name__ == "__main__":
    main()
