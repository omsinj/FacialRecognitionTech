"""
create_unknown_split.py

Purpose
-------
Derive an additional "unknown" split from the existing embeddings table by
re-labelling a subset of test identities as UNKNOWN. This is used to simulate
realistic impostor attempts for open-set evaluation and threshold calibration.

Why this is needed
------------------
The original pipeline only distinguishes:
    - 'enroll' : identities that form the gallery
    - 'test'   : held-out samples for closed-set evaluation

However, open-set access control must also handle *unseen* individuals.
To emulate this, we:
    1. Randomly select N identities that currently only appear in the 'test' split.
    2. Re-label all their test samples as 'unknown'.
    3. Save the modified table as a separate embeddings file.

Inputs
------
data/meta/embeddings.parquet
    Produced by build_embeddings.py.
    Must contain at least the columns:
        - identity
        - split
        - (and the 512-D embedding columns, which are preserved as-is)

Outputs
-------
data/meta/embeddings_with_unknown.parquet
    Same structure as embeddings.parquet, but with:
        - some rows moved from split == 'test' to split == 'unknown'.

Usage
-----
python scripts/create_unknown_split.py
    - by default, selects 10 identities to become 'unknown'.

You can also import and call:

    from scripts.create_unknown_split import create_unknown_split
    df_mod = create_unknown_split(n_unknown=15, seed=123)

Notes
-----
- The selection is identity-level: all test samples of a chosen identity
  become 'unknown'.
- Randomness is controlled by a NumPy Generator and a fixed seed for
  reproducibility.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ----------------------------------------------------------------------
# Project paths (derived relative to this script)
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
META_DIR = DATA_DIR / "meta"
EMB_PATH = META_DIR / "embeddings.parquet"


def create_unknown_split(n_unknown: int = 10, seed: int = 42) -> pd.DataFrame:
    """
    Re-label a subset of test identities as UNKNOWN in the embeddings table.

    Parameters
    ----------
    n_unknown : int, optional
        Number of distinct identities (from the 'test' split) to move to the
        'unknown' split. Default is 10.
    seed : int, optional
        Random seed for NumPy's Generator to ensure reproducible identity
        selection. Default is 42.

    Returns
    -------
    df_modified : pandas.DataFrame
        The modified embeddings DataFrame with an additional 'unknown' split.
        Also written to `data/meta/embeddings_with_unknown.parquet`.

    Behaviour
    ---------
    1. Load existing embeddings.
    2. Identify all identities that appear in split == 'test'.
    3. Randomly select `n_unknown` of these identities.
    4. Set split = 'unknown' for all rows belonging to those identities
       that were previously in 'test'.
    5. Report:
        - overall split counts
        - number of unique identities per split
    6. Save the modified table.
    """
    # ------------------------------------------------------------------
    # 1) Load embeddings (must already exist from build_embeddings.py)
    # ------------------------------------------------------------------
    df = pd.read_parquet(EMB_PATH)

    # ------------------------------------------------------------------
    # 2) Enumerate identities currently in the test split
    # ------------------------------------------------------------------
    test_identities = df[df["split"] == "test"]["identity"].unique()
    print(f"Total test identities: {len(test_identities)}")

    if n_unknown > len(test_identities):
        raise ValueError(
            f"Requested n_unknown={n_unknown}, "
            f"but only {len(test_identities)} test identities are available."
        )

    # ------------------------------------------------------------------
    # 3) Randomly select a subset of test identities to label as UNKNOWN
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    unknown_ids = rng.choice(test_identities, size=n_unknown, replace=False)
    print(
        f"Selected {len(unknown_ids)} identities for unknown split: "
        f"{sorted(unknown_ids)}"
    )

    # ------------------------------------------------------------------
    # 4) Apply the re-labelling: test → unknown for chosen identities
    # ------------------------------------------------------------------
    df_modified = df.copy()
    mask = (df_modified["split"] == "test") & (df_modified["identity"].isin(unknown_ids))
    df_modified.loc[mask, "split"] = "unknown"

    # ------------------------------------------------------------------
    # 5) Report new distribution of splits and identities
    # ------------------------------------------------------------------
    print("\nNew split counts:")
    print(df_modified["split"].value_counts())

    print("\nUnique identities per split:")
    print(df_modified.groupby("split")["identity"].nunique())

    # ------------------------------------------------------------------
    # 6) Persist the modified embeddings table
    # ------------------------------------------------------------------
    output_path = META_DIR / "embeddings_with_unknown.parquet"
    df_modified.to_parquet(output_path)
    print(f"\nSaved to: {output_path}")

    return df_modified


if __name__ == "__main__":
    # Default CLI behaviour: promote 10 test identities to UNKNOWN with a fixed seed
    create_unknown_split(n_unknown=10)
