# scripts/eval_shuffle_baseline.py
"""
eval_shuffle_baseline.py

Purpose
-------
Sanity-check that the classifier is learning meaningful structure
from the embeddings, rather than exploiting artefacts or label
leakage.

The idea:
    - Evaluate the classifier on the true labels for the test split.
    - Randomly shuffle the labels and re-evaluate using the same
      predictions.
    - A well-behaved system should show:
        * High accuracy with true labels.
        * Accuracy collapsing towards chance level with shuffled labels.

If accuracy remains high even after shuffling, this is a red flag
that something is wrong with the data pipeline (e.g. leakage,
duplicate labels, or an evaluation bug).
"""

import os

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from fr_utils.config import META_DIR, MODELS_DIR


def main():
    """
    Run the shuffle-baseline sanity check on the test split.

    Steps
    -----
    1. Load embeddings.parquet from META_DIR.
    2. Filter to rows where split == 'test'.
    3. Build the feature matrix X by dropping metadata columns
       ('identity', 'split', 'image_id').
    4. Extract the true identity labels y.
    5. Create a shuffled version of the labels y_shuf using a fixed
       random seed for reproducibility.
    6. Load the trained classifier from MODELS_DIR.
    7. Compute predictions on X.
    8. Report:
           - Accuracy with true labels (should be high).
           - Accuracy with shuffled labels (should be near chance).

    Interpretation
    -------------
    - If "Sanity (true labels)" is high and
      "Sanity (shuffled labels)" is ~0,
      then the model is genuinely discriminative.

    - If both accuracies are similar or both high, investigate:
        * label leakage between train/test,
        * duplicated samples,
        * or bugs in how labels are assigned.
    """
    # Load all stored embeddings and metadata
    emb_path = os.path.join(META_DIR, "embeddings.parquet")
    df = pd.read_parquet(emb_path)

    # Restrict to the test split only
    test = df[df["split"] == "test"].copy()

    # Feature matrix: drop non-embedding metadata columns
    # (identity, split, image_id). Remaining columns are embedding dims.
    X = test.drop(columns=["identity", "split", "image_id"]).to_numpy(dtype=np.float32)

    # True labels as strings (consistent with training)
    y = test["identity"].astype(str).to_numpy()

    # Shuffled labels for the baseline: same values, different order
    # random_state is fixed for reproducibility across runs.
    y_shuf = shuffle(y, random_state=123)

    # Load trained classifier from disk
    clf_path = os.path.join(MODELS_DIR, "classifier.joblib")
    clf = load(clf_path)

    # Closed-set predictions on the test features
    y_pred = clf.predict(X)

    # Accuracy with correct labels
    acc_true = accuracy_score(y, y_pred)

    # Accuracy when pairing the same predictions with randomly permuted labels
    acc_shuf = accuracy_score(y_shuf, y_pred)

    # Report
    print("Sanity (true labels):     ", acc_true)
    print("Sanity (shuffled labels): ", acc_shuf)


if __name__ == "__main__":
    main()
