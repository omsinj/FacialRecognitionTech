"""
build_embeddings.py

Purpose
-------
Run FaceNet inference on all aligned facial crops and persist their 512-D embedding
vectors to a single `embeddings.parquet` file. These embeddings are later consumed by:
 - classifier training (KNN)
 - centroid computation
 - FAR/FRR evaluation
 - open-set evaluation
 - robustness evaluation
 - the access-control application

Inputs (pre-requisite)
----------------------
data/meta/subset_aligned.csv
    Produced by preprocess_align.py using the 50×20 subset of CelebA.
    Contains at least:
        image_id : filename of aligned crop
        identity : string identity label
        split    : 'enroll' or 'test'

Associated directories (from fr_utils.config):
    ALIGNED_DIR : contains final 160×160 aligned crops
    META_DIR    : metadata + evaluation artifacts

Output
-------
data/meta/embeddings.parquet
    Tabular file containing:
        columns 0..511    : embedding dimensions (float32)
        identity          : identity label (string)
        split             : enrol/test flag
        image_id          : filename reference

Execution
---------
python scripts/build_embeddings.py

Notes
-----
 - Images that fail to load or embed are skipped silently.
 - Embeddings are stored row-wise as NumPy float32.
"""

import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from fr_utils.config import ALIGNED_DIR, META_DIR
from fr_utils.facenet_backend import embed_image


def main():
    """
    Iterate through all aligned images, obtain a 512-D embedding per image, and
    assemble a DataFrame combining embeddings and metadata.

    This forms the backbone of most downstream evaluation: every experiment
    references this parquet instead of re-running expensive inference.
    """
    # ------------------------------------------------------------------
    # 0) Sanity: ensure alignment step was performed
    # ------------------------------------------------------------------
    subset_path = os.path.join(META_DIR, "subset_aligned.csv")
    if not os.path.exists(subset_path):
        raise SystemExit(
            "Missing data/meta/subset_aligned.csv — run preprocess_align first."
        )

    subset = pd.read_csv(subset_path)

    # Lists to accumulate embedding rows and metadata
    embs, labels, splits, files = [], [], [], []

    # ------------------------------------------------------------------
    # 1) Loop over each aligned face crop; extract FaceNet embedding
    # ------------------------------------------------------------------
    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Embedding faces"):
        img_path = os.path.join(ALIGNED_DIR, row["image_id"])
        bgr = cv2.imread(img_path)

        # Skip unreadable or missing images
        if bgr is None:
            continue

        try:
            # embed_image returns a 512-D float vector
            emb = embed_image(bgr)
            embs.append(emb)

            # Keep original metadata with the embedding row
            labels.append(str(row["identity"]))
            splits.append(row["split"])
            files.append(row["image_id"])

        except Exception:
            # If FaceNet inference fails for a specific file, skip but continue.
            continue

    # ------------------------------------------------------------------
    # 2) Convert list-of-vectors to a single matrix and wrap in a DataFrame
    # ------------------------------------------------------------------
    X = np.vstack(embs).astype(np.float32)

    df = pd.DataFrame(X)  # columns become stringified indices "0","1",...,"511"
    df["identity"] = labels
    df["split"] = splits
    df["image_id"] = files

    # ------------------------------------------------------------------
    # 3) Persist to parquet for fast loading by downstream tasks
    # ------------------------------------------------------------------
    out_path = os.path.join(META_DIR, "embeddings.parquet")
    df.to_parquet(out_path)

    print(f"Saved embeddings to {out_path} with shape: {X.shape}")


if __name__ == "__main__":
    main()
