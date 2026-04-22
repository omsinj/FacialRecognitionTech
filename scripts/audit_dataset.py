# scripts/audit_dataset.py

"""
audit_dataset.py

Purpose
-------
This diagnostic script checks for data contamination and leakage between the
enrolment and test splits used in the facial recognition experiments.

Specifically, it answers three critical questions:

1. Are any image IDs (filenames) present in BOTH the enrol and test splits?
   - This would indicate a direct split leak and artificially inflate performance.

2. Are there any *content-identical* images across splits?
   - Detected using SHA-1 hashes of the aligned images in data/aligned.
   - This catches the case where the same image is duplicated under different metadata.

3. Are there any *near-duplicate* images across splits?
   - Detected via a perceptual hash (pHash) and Hamming distance.
   - This captures images that are visually extremely similar (e.g. tiny changes,
     compression differences), which can also bias evaluation.

This script is intended as a one-off or periodic audit tool to support the integrity
of evaluation results and is safe to run multiple times.
"""

import os, sys, hashlib, cv2, numpy as np, pandas as pd
from pathlib import Path

from fr_utils.config import META_DIR, ALIGNED_DIR


def file_sha1(path):
    """
    Compute the SHA-1 checksum of a file.

    Parameters
    ----------
    path : str or Path
        Filesystem path to the image file.

    Returns
    -------
    str
        Hexadecimal SHA-1 digest of the file contents.

    Notes
    -----
    SHA-1 is used here only as a *fingerprint* to detect exact duplicates.
    This is not a cryptographic security operation; it is purely for dataset
    integrity checking.
    """
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def phash(img, hash_size=16):
    """
    Compute a simple perceptual hash (pHash) for an image.

    Parameters
    ----------
    img : np.ndarray
        Input image in BGR format (as loaded by OpenCV).
    hash_size : int, optional
        Size of the low-frequency DCT block to use for hashing. The effective
        bit-length of the hash will be `hash_size * hash_size` bits.

    Returns
    -------
    str
        A binary string (e.g. '010101...') representing the perceptual hash.

    Method
    ------
    1. Convert to grayscale.
    2. Resize to a larger square (hash_size * 4) to ensure enough frequency detail.
    3. Compute the 2D Discrete Cosine Transform (DCT).
    4. Take the top-left `hash_size x hash_size` block (low-frequency components).
    5. Threshold coefficients at their median to generate a binary pattern.

    Rationale
    ---------
    Unlike SHA-1, this hash is designed to be *robust* to tiny visual changes,
    such as slight compression differences or minor transformations, making it
    suitable for detecting near-duplicates by Hamming distance.
    """
    # Convert BGR → grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Downsample aggressively while keeping enough structure for DCT
    img = cv2.resize(img, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA)

    img = np.float32(img)

    # 2D DCT over the full resized image
    dct = cv2.dct(img)

    # Take only the lowest-frequency coefficients
    low = dct[:hash_size, :hash_size]

    # Use the median as a threshold to get a robust binary pattern
    med = np.median(low)

    # Flatten and convert to a binary string based on the median threshold
    return "".join("1" if x > med else "0" for x in low.flatten())


def main():
    """
    Run the dataset audit.

    Workflow
    --------
    1. Load embeddings metadata from `embeddings.parquet` in META_DIR.
       - Require at minimum `image_id` and `split` columns.
       - Derive the list of enrol and test image IDs.

    2. Check for direct leakage:
       - Compute set intersection between enrol and test image IDs.
       - Any overlap is a structural bug in the split generation.

    3. Build content-based hashes for each split:
       - SHA-1 over aligned image files in ALIGNED_DIR for exact duplicates.
       - Perceptual hashes (pHash) for near-duplicate detection.

    4. Detect:
       - Exact duplicate files across splits via SHA-1 collisions.
       - Near-duplicate pairs across splits via pHash Hamming distance <= 4.

    Output
    ------
    - Prints summary statistics to stdout.
    - Does not modify any files or metadata.

    This script is intentionally read-only and safe to run in production or
    evaluation environments.
    """

    # ------------------------------------------------------------------
    # Load embedding metadata and validate schema
    # ------------------------------------------------------------------
    df = pd.read_parquet(Path(META_DIR) / "embeddings.parquet")
    assert {"image_id", "split"}.issubset(df.columns), \
        "embeddings.parquet must have image_id and split columns"

    # Extract enrolled and test image IDs as Python lists
    enroll = df[df["split"] == "enroll"]["image_id"].tolist()
    test   = df[df["split"] == "test"]["image_id"].tolist()

    # ------------------------------------------------------------------
    # 1) Structural leakage: same image_id present in both splits
    # ------------------------------------------------------------------
    dup_ids = set(enroll).intersection(test)
    print(f"Image IDs present in BOTH splits: {len(dup_ids)}")
    if dup_ids:
        # In a clean pipeline this should be zero; any non-zero value is a red flag.
        print("Example overlapping IDs (first 10):", list(dup_ids)[:10])

    # ------------------------------------------------------------------
    # 2) Content-based duplicates across splits
    #
    # We now ignore IDs and look at the actual image content to catch:
    #  - Same image with same filename but wrong split
    #  - Same image duplicated under a different filename in another split
    # ------------------------------------------------------------------

    def build_maps(names):
        """
        Build SHA-1 and pHash maps for a list of filenames.

        Parameters
        ----------
        names : list[str]
            Filenames (image_id values) to inspect under ALIGNED_DIR.

        Returns
        -------
        tuple[dict, dict]
            sha_map  : { image_id -> sha1_hex }
            phash_map: { image_id -> pHash_binary_string }
        """
        sha_map, phash_map = {}, {}

        for name in names:
            p = Path(ALIGNED_DIR) / name
            if not p.exists():
                # Aligned image missing; this may have been flagged already
                # by other integrity tools (e.g. a_sanity_check.py).
                continue

            # Exact file hash (byte-identical)
            sha = file_sha1(p)
            sha_map[name] = sha

            # Perceptual hash for near-duplicate detection
            bgr = cv2.imread(str(p))
            if bgr is None:
                # Corrupted or unreadable image; skip from pHash analysis
                continue
            ph = phash(bgr)
            phash_map[name] = ph

        return sha_map, phash_map

    sha_enroll, ph_enroll = build_maps(enroll)
    sha_test,   ph_test   = build_maps(test)

    # ------------------------------------------------------------------
    # Exact duplicate detection using SHA-1
    # ------------------------------------------------------------------
    # Invert the mapping to: sha1 -> [filenames] so we can see collisions.
    inv_sha_en = {}
    for k, v in sha_enroll.items():
        inv_sha_en.setdefault(v, []).append(k)

    inv_sha_te = {}
    for k, v in sha_test.items():
        inv_sha_te.setdefault(v, []).append(k)

    # Any SHA-1 hash appearing in both inverted maps indicates that at least
    # two files (one in each split) are *bit-for-bit identical*.
    exact_collisions = set(inv_sha_en.keys()).intersection(inv_sha_te.keys())
    print(f"Exact file duplicates across splits (SHA1): {len(exact_collisions)}")
    if exact_collisions:
        # Show a small sample of duplicate SHA1 groups for debugging
        sample = list(exact_collisions)[:5]
        print("Sample duplicate SHA1 groups (enroll|test):")
        for sha in sample:
            print("  SHA1:", sha)
            print("    enroll:", inv_sha_en.get(sha, []))
            print("    test  :", inv_sha_te.get(sha, []))

    # ------------------------------------------------------------------
    # Near-duplicate detection using perceptual hash similarity (pHash)
    # ------------------------------------------------------------------
    def hamming(a, b):
        """
        Compute the Hamming distance between two equal-length strings.

        Parameters
        ----------
        a, b : str
            Strings representing binary hash codes, e.g. '0101...'.

        Returns
        -------
        int
            Number of positions at which the corresponding bits differ.

        Notes
        -----
        A small Hamming distance (e.g. <= 4 for a 256-bit hash) indicates that
        the two images are visually extremely similar and may be near-duplicates.
        """
        return sum(c1 != c2 for c1, c2 in zip(a, b))

    near = []

    # Brute-force comparison: for the current dataset scale (50 identities,
    # ~1000 images total), an O(N^2) comparison between splits is acceptable.
    for k_en, h_en in ph_enroll.items():
        for k_te, h_te in ph_test.items():
            if hamming(h_en, h_te) <= 4:  # very close in perceptual space
                near.append((k_en, k_te))

    print(f"Near-duplicate pairs across splits (pHash<=4): {len(near)}")
    if near[:10]:
        print("Sample near-duplicate pairs:", near[:10])


if __name__ == "__main__":
    # Entry point: run the audit with default parameters.
    # Typical usage:
    #   python scripts/audit_dataset.py
    main()
