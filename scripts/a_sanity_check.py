"""
a_sanity_check.py

This script performs a lightweight integrity audit of the CelebA subset used by the
project. It verifies that metadata, raw image files, and aligned images exist and are
readable, and optionally attempts re-alignment of missing aligned images.

This script is intentionally self-contained and does not depend on the full runtime,
allowing it to be executed as an independent diagnostic utility.
"""

import os
import csv
import cv2
import argparse
import pandas as pd
from collections import Counter


# ---------------------------------------------------------------------------
# Define project paths relative to the repository root. These are deliberately
# resolved locally rather than imported from fr_utils.config to ensure the script
# can be executed in isolation as a standalone integrity check.
# ---------------------------------------------------------------------------

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
ALIGNED_DIR = os.path.join(DATA_DIR, "aligned")
META_DIR = os.path.join(DATA_DIR, "meta")


def main(fix_missing_align: bool = False, sample_check: int = 25):
    """
    Execute data integrity validation.

    Parameters
    ----------
    fix_missing_align : bool
        If True, attempt to re-align any missing aligned images using MTCNN.
    sample_check : int
        Number of aligned images to sample and verify as readable.

    Behaviour
    ---------
    - Checks metadata consistency in subset.csv
    - Confirms that raw and aligned images exist
    - Optionally attempts re-alignment using the same alignment pipeline
    - Reports missing or unreadable images
    - Writes a brief machine-readable integrity summary (CSV)

    Returns
    -------
    None
    """

    # ---------------- Metadata validation -----------------------------------
    subset_csv = os.path.join(META_DIR, "subset.csv")
    assert os.path.exists(subset_csv), f"subset.csv not found at {subset_csv}"

    df = pd.read_csv(subset_csv)
    assert {"image_id","identity","split"}.issubset(df.columns), \
        "subset.csv missing required columns"


    # ---------------- Structural statistics ---------------------------------
    n_rows = len(df)
    ids = sorted(df["identity"].unique().tolist())
    splits = Counter(df["split"].tolist())
    per_id = df.groupby("identity")["image_id"].count().to_dict()


    # ---------------- Presence of raw/aligned files --------------------------
    missing_raw = []
    missing_aligned = []

    for _, r in df.iterrows():
        raw_p = os.path.join(RAW_DIR, r["image_id"])
        if not os.path.exists(raw_p):
            missing_raw.append(r["image_id"])

        aligned_p = os.path.join(ALIGNED_DIR, r["image_id"])
        if not os.path.exists(aligned_p):
            missing_aligned.append(r["image_id"])


    # ---------------- Optional re-alignment of missing files -----------------
    fixed = 0
    if fix_missing_align and missing_aligned:
        try:
            # Uses the same FaceNet/MTCNN alignment used during preprocessing
            from fr_utils.facenet_backend import align_face
        except Exception as e:
            print("Could not import fr_utils.facenet_backend:", e)
            fix_missing_align = False

        if fix_missing_align:
            os.makedirs(ALIGNED_DIR, exist_ok=True)

            for fname in missing_aligned:
                raw_p = os.path.join(RAW_DIR, fname)
                if not os.path.exists(raw_p):
                    continue

                bgr = cv2.imread(raw_p)
                if bgr is None:
                    continue

                # Attempt alignment, ignoring failures gracefully
                try:
                    rgb = align_face(bgr)
                    cv2.imwrite(
                        os.path.join(ALIGNED_DIR, fname),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    )
                    fixed += 1
                except Exception:
                    pass


    # ---------------- Lightweight image readability check --------------------
    unreadable = []
    for fname in df["image_id"].head(sample_check):
        path = os.path.join(ALIGNED_DIR, fname)
        if cv2.imread(path) is None:
            unreadable.append(fname)


    # ---------------- Human-readable summary ---------------------------------
    print("=== DATA SANITY REPORT ===")
    print(f"subset rows: {n_rows}")
    print(f"identities: {len(ids)} -> {ids[:10]}{' ...' if len(ids)>10 else ''}")
    print(f"splits: {dict(splits)}")
    print(f"per-identity counts (first 10): {dict(list(per_id.items())[:10])}")
    print(f"missing in raw: {len(missing_raw)}")
    print(f"missing in aligned: {len(missing_aligned)}")

    if fix_missing_align:
        print(f"realigned (fixed): {fixed}")

    print(f"unreadable aligned (sample {sample_check}): {len(unreadable)}")


    # ---------------- CSV summary for reproducibility ------------------------
    out_csv = os.path.join(META_DIR, "sanity_summary.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["subset_rows", n_rows])
        w.writerow(["n_identities", len(ids)])
        w.writerow(["missing_raw", len(missing_raw)])
        w.writerow(["missing_aligned", len(missing_aligned)])
        w.writerow(["unreadable_aligned_sample", len(unreadable)])

    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    """
    Command-line entry point.
    Typical usage:
        python a_sanity_check.py
        python a_sanity_check.py --fix-missing-align
        python a_sanity_check.py --sample-check 50
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("--fix-missing-align", action="store_true",
        help="Attempt MTCNN re-alignment for missing aligned images.")
    ap.add_argument("--sample-check", type=int, default=25,
        help="Number of aligned images to verify for readability.")

    args = ap.parse_args()
    main(fix_missing_align=args.fix_missing_align,
         sample_check=args.sample_check)
