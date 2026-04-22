"""
build_centroids.py

Purpose
-------
Compute per-identity centroids in the FaceNet embedding space, along with a simple
"intra-class radius" for each identity.

These centroids and radii are saved to `META_DIR/centroids.npz` and are used by:
- Open-set decision functions (distance to class centroid + thresholding).
- Evaluation scripts that need a compact representation of each enrolled identity.

Concepts
--------
- Embeddings:
    512-dimensional FaceNet feature vectors, already stored in embeddings.parquet.
- Centroid:
    L2-normalised mean embedding for all enrolment samples of a given identity.
- Radius:
    Mean cosine *distance* between each enrolment embedding and the identity's centroid.
    This provides a crude measure of how "spread out" the identity is in embedding space.
"""

import os
import numpy as np
import pandas as pd

from fr_utils.config import META_DIR

# Output file for centroid data (labels, centroids, radii)
OUT_PATH = os.path.join(META_DIR, "centroids.npz")


def main():
    """
    Load embedding metadata, compute per-identity centroids and radii, and persist them.

    Workflow
    --------
    1. Load `embeddings.parquet` from META_DIR.
       - Expect columns:
         - "identity": class label (string or int)
         - "split": 'enroll' / 'test'
         - "0"..."511": embedding dimensions (stored as strings in Parquet)

    2. Filter to the enrolment split only (split == 'enroll').
       - Centroids should be derived solely from the gallery / known identities.

    3. For each identity in the enrolment set:
       - Extract its embeddings into a matrix X of shape [n_samples, n_features].
       - L2-normalise each row (embedding) to reflect cosine-space behaviour.
       - Compute the mean vector μ over the normalised rows.
       - L2-normalise μ to unit length.
       - Compute cosine distances for each sample to μ:
           cos_sim_i = x_i · μ
           cos_dist_i = 1 - cos_sim_i
         and take the mean distance as the class "radius".

    4. Stack all centroids, labels, and radii into arrays and save to OUT_PATH
       in compressed .npz format.

    Output structure
    ----------------
    OUT_PATH (centroids.npz) contains:
        - labels    : np.ndarray[str]   , shape [n_classes]
        - centroids : np.ndarray[float32], shape [n_classes, n_features]
        - radii     : np.ndarray[float32], shape [n_classes]

    This file is later loaded by:
        - open-set evaluation scripts
        - prototype access-control application
    """

    # ------------------------------------------------------------------
    # 1) Load embeddings parquet
    # ------------------------------------------------------------------
    embeddings_path = os.path.join(META_DIR, "embeddings.parquet")
    df = pd.read_parquet(embeddings_path)

    # Feature columns are numeric embeddings: "0", "1", ..., "511"
    # We detect them by checking if the column name is a digit string.
    feat_cols = [c for c in df.columns if str(c).isdigit()]

    # Restrict to enrolment images only; test images are not used to build centroids.
    enrolled = df[df["split"] == "enroll"].copy()

    # Prepare containers for centroid data
    labels = []    # identity labels (stringified)
    centroids = [] # centroid vectors (np.ndarray)
    radii = []     # mean intra-class cosine distance per identity

    # ------------------------------------------------------------------
    # 2) Loop over identities and build centroids + radii
    # ------------------------------------------------------------------
    for identity, g in enrolled.groupby("identity"):
        # Extract embeddings for this identity as a dense NumPy array
        X = g[feat_cols].to_numpy(dtype=np.float32)

        # L2-normalise each embedding vector (row-wise) to unit length.
        # This matches the assumptions used when comparing embeddings
        # with cosine similarity.
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        # Compute the mean embedding (centroid) over all normalised samples
        mu = Xn.mean(axis=0)

        # Normalise centroid as well to unit length
        mu = mu / (np.linalg.norm(mu) + 1e-8)

        # ------------------------------------------------------------------
        #  Radius: average cosine distance of samples to their class centroid
        #
        #  cos_sim_i  = x_i · mu           (because both are unit-length)
        #  cos_dist_i = 1 - cos_sim_i
        #
        #  Intuition:
        #    - Small radius  -> tightly clustered class, low intra-class variance.
        #    - Large radius  -> more variable appearances for that identity.
        # ------------------------------------------------------------------
        cos_sim = (Xn @ mu)          # shape: [n_samples]
        dists = 1.0 - cos_sim        # cosine distance
        r = float(dists.mean())      # scalar radius for this identity

        labels.append(str(identity))
        centroids.append(mu.astype(np.float32))
        radii.append(r)

    # Convert Python lists to NumPy arrays with explicit dtypes
    labels = np.array(labels, dtype="<U16")     # fixed-width unicode strings
    centroids = np.stack(centroids).astype(np.float32)
    radii = np.array(radii, dtype=np.float32)

    # ------------------------------------------------------------------
    # 3) Persist centroid data to compressed NPZ
    # ------------------------------------------------------------------
    np.savez_compressed(OUT_PATH, labels=labels, centroids=centroids, radii=radii)
    print(f"Saved centroids: {OUT_PATH} | classes={len(labels)}")


if __name__ == "__main__":
    # Allow running as a standalone script:
    #   python scripts/build_centroids.py
    main()
