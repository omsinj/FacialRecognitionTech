# scripts/preprocess_align.py
"""
preprocess_align.py

Purpose
-------
Run face detection + alignment on the raw CelebA images defined in subset.csv
and write standardised 160×160 face crops into data/aligned.

This script:
    - Reads the manifest data/meta/subset.csv produced by make_subset.py
    - For each listed image:
        * loads the raw image from data/raw
        * applies MTCNN-based alignment via fr_utils.facenet_backend.align_face
        * saves a cropped/aligned face into data/aligned with the same filename
    - Writes data/meta/subset_aligned.csv containing only those rows for which
      alignment succeeded.

Downstream scripts (build_embeddings.py, evaluation, etc.) use subset_aligned.csv
as the ground truth list of available aligned faces.
"""

import os
import pandas as pd
import cv2
from tqdm import tqdm

from fr_utils.config import RAW_DIR, ALIGNED_DIR, META_DIR
from fr_utils.facenet_backend import align_face


def main() -> None:
    """
    Align faces for all images listed in subset.csv and build subset_aligned.csv.

    Flow
    ----
    1. Load data/meta/subset.csv:
           columns: image_id, identity, split
    2. For each row:
           - read data/raw/<image_id>
           - detect + align the primary face using align_face(...)
           - write aligned crop to data/aligned/<image_id>
    3. Collect only successful rows in subset_aligned.csv so downstream code
       never attempts to embed missing or failed images.
    """
    # Path to the subset manifest created by make_subset.py
    subset_path = os.path.join(META_DIR, "subset.csv")
    subset = pd.read_csv(subset_path)

    # Ensure the aligned output directory exists
    os.makedirs(ALIGNED_DIR, exist_ok=True)

    kept = []  # rows for which alignment succeeded

    # iterate over all requested images and align them
    for _, row in tqdm(subset.iterrows(), total=len(subset)):
        img_path = os.path.join(RAW_DIR, row["image_id"])

        # Load raw BGR image from disk
        bgr = cv2.imread(img_path)
        if bgr is None:
            # File missing or unreadable; silently skip but do not crash the batch
            continue

        try:
            # align_face performs detection + alignment and returns RGB crop
            # expected shape ~ (160, 160, 3)
            rgb = align_face(bgr)

            # Save back as BGR into the aligned directory using the same filename
            out_path = os.path.join(ALIGNED_DIR, row["image_id"])
            cv2.imwrite(out_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            # keep the original metadata row so we know this image is usable downstream
            kept.append(row)
        except Exception:
            # Any detection/alignment failure is ignored for robustness.
            # The image will simply be omitted from subset_aligned.csv.
            continue

    # ------------------------------------------------------------------
    # Persist aligned manifest: only those rows that survived alignment
    # ------------------------------------------------------------------
    if kept:
        df_kept = pd.DataFrame(kept)
        out_csv = os.path.join(META_DIR, "subset_aligned.csv")
        df_kept.to_csv(out_csv, index=False)
        print(
            f"Aligned {len(df_kept)} / {len(subset)} images "
            f"-> {out_csv}"
        )
    else:
        # If nothing aligned, something is fundamentally wrong
        print("No faces aligned; check your data and detector settings.")


if __name__ == "__main__":
    main()
