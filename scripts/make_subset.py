# scripts/make_subset.py
"""
make_subset.py

Purpose
-------
Creates a deterministic subset definition (subset.csv) for the CelebA dataset
containing a controlled number of identities and per-identity images,
split into "enroll" and "test" subsets.

This script:
    - Reads CelebA identity_CelebA.txt annotation file.
    - Filters identities that have enough images.
    - Selects the first N identities deterministically (no randomness).
    - Assigns each identity:
          * first <enroll> images to the enroll split
          * next  <test>   images to the test split
    - Writes a clean manifest CSV: data/meta/subset.csv

This subset is subsequently used by the alignment and embedding pipelines.
"""

import argparse, os, random
import pandas as pd

from tqdm import tqdm
from fr_utils.config import RAW_DIR, META_DIR, SEED


# --------------------------------------------------------------------------
# Fix RNG for reproducible behaviour (although script selection is deterministic)
# --------------------------------------------------------------------------
random.seed(SEED)


def load_identity_map(identity_txt_path: str) -> pd.DataFrame:
    """
    Parse CelebA annotation file (identity_CelebA.txt), formatted:

        <image_name>   <identity_id>

    into a DataFrame with columns:
        - image_id
        - identity (integer)

    Parameters
    ----------
    identity_txt_path : str
        Absolute path to the CelebA identity annotation.

    Returns
    -------
    pd.DataFrame
        DataFrame(images, identities)
    """
    rows = []
    with open(identity_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img, pid = line.split()
            rows.append((img, int(pid)))

    return pd.DataFrame(rows, columns=["image_id", "identity"])


# --------------------------------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------------------------------
def main(identities: int, enroll: int, test: int):
    """
    Build a deterministic <subset.csv> manifest for CelebA, selecting the
    first <identities> identities having at least (enroll + test) images.

    Parameters
    ----------
    identities : int
        Number of distinct person IDs to include in the subset.
    enroll : int
        Number of reference images per identity (enrolment).
    test : int
        Number of evaluation images per identity (verification/test).

    Output
    ------
    Writes:
        data/meta/subset.csv
    containing rows:
        image_id, identity, split
    """
    id_path = os.path.join(META_DIR, "identity_CelebA.txt")

    if not os.path.exists(id_path):
        raise SystemExit(
            f"Missing {id_path}\n"
            "Download identity_CelebA.txt (CelebA annotation) and place it in data/meta/"
        )

    # Load full annotation mapping: image → identity
    df_ids = load_identity_map(id_path)

    # Count available images per identity
    counts = df_ids["identity"].value_counts()

    # Keep identities with sufficient images for requested splits
    valid_ids = counts[counts >= (enroll + test)].index.tolist()

    # Deterministic selection: sorted identities, take first N
    chosen_ids = sorted(valid_ids)[:identities]
    df_ids = df_ids[df_ids["identity"].isin(chosen_ids)]

    # ------------------------------------------------------------------
    # Construct row-level records (image, identity, split)
    # ------------------------------------------------------------------
    rows = []
    for pid in chosen_ids:
        imgs = (
            df_ids[df_ids["identity"] == pid]["image_id"]
            .sort_values()                 # ensure deterministic ordering
            .tolist()
        )
        chosen = imgs[: (enroll + test)]
        enroll_imgs = chosen[:enroll]
        test_imgs   = chosen[enroll:(enroll + test)]

        # tag assigned rows
        for fn in enroll_imgs:
            rows.append({"image_id": fn, "identity": str(pid), "split": "enroll"})
        for fn in test_imgs:
            rows.append({"image_id": fn, "identity": str(pid), "split": "test"})

    subset = pd.DataFrame(rows)

    # ensure output dirs exist (raw may be still empty at this stage)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)

    # Write final manifest
    out_csv = os.path.join(META_DIR, "subset.csv")
    subset.to_csv(out_csv, index=False)

    print(
        f"Saved {out_csv} with {len(subset)} rows "
        f"across {len(chosen_ids)} identities."
    )


# --------------------------------------------------------------------------
# CLI ENTRY POINT
# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Create deterministic CelebA subset manifest.")
    ap.add_argument("--identities", type=int, default=50, help="Number of identities to include.")
    ap.add_argument("--enroll",     type=int, default=10, help="Enrollment images per identity.")
    ap.add_argument("--test",       type=int, default=10, help="Test images per identity.")
    args = ap.parse_args()

    main(args.identities, args.enroll, args.test)
