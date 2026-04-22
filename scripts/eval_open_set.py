# scripts/eval_open_set.py
"""
eval_open_set.py

Purpose
-------
Evaluate the access-control system in an open-set setting.

Given:
    - Stored embeddings (including an "unknown" split).
    - A trained KNN classifier + label encoder.
    - Optional probability and distance thresholds.

This script:
    - Treats the classifier as closed-set for the initial prediction.
    - Applies an open-set rejection rule based on:
        * predicted class probability (tau), and
        * optional distance-to-centroid (dist_threshold).
    - Computes:
        * Closed-set Top-1 accuracy.
        * Open-set Top-1 accuracy on accepted examples only.
        * Overall rejection rate.
        * Fraction of misclassifications that are successfully rejected as UNKNOWN.
        * Macro-precision/recall/F1 restricted to accepted samples.
    - Optionally sweeps ranges of thresholds and saves the results to CSV.

Typical usage
-------------
Single operating point:

    python scripts/eval_open_set.py --tau 0.80 --dist 0.90

Sweep probability and distance thresholds and log results:

    python scripts/eval_open_set.py \
        --sweep_prob 0.50:0.95:0.05 \
        --sweep_dist 0.70:1.20:0.05 \
        --report data/meta/evaluation/open_set_sweep.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------------------------------------------------------------
# Project directories (resolved relative to this script)
# ----------------------------------------------------------------------
# We avoid importing fr_utils.config here so this script can be run
# directly or via `python -m` without requiring package installation.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
META_DIR = DATA_DIR / "meta"
MODELS_DIR = ROOT / "models"

# Embeddings file with an explicit 'unknown' split created by
# scripts/create_unknown_split.py.
EMB_PATH = META_DIR / "embeddings_with_unknown.parquet"


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def _find_embedding_columns(df: pd.DataFrame) -> list[str]:
    """
    Identify embedding columns in the embeddings DataFrame.

    This function is robust to two common naming patterns:
        1) Numeric string columns: '0', '1', ..., '511'
        2) Prefixed columns: 'e0', 'e1', ..., 'e511'

    It returns the discovered embedding column names in ascending
    numeric order, which are then used to build the feature matrix.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame loaded from the embeddings parquet.

    Returns
    -------
    list[str]
        Sorted list of column names that correspond to embedding
        dimensions.

    Raises
    ------
    RuntimeError
        If no suitable embedding columns are found.
    """
    # Case 1: purely numeric column labels (e.g., "0", "1", ...)
    num_cols = [c for c in df.columns if isinstance(c, str) and c.isdigit()]
    if len(num_cols) >= 128:  # heuristic: expect many dims for embeddings
        num_cols_sorted = sorted(num_cols, key=lambda x: int(x))
        return num_cols_sorted

    # Case 2: 'e0', 'e1', ... style column labels
    ecols = [c for c in df.columns if isinstance(c, str) and c.startswith("e")]
    if len(ecols) >= 128:
        ecols_sorted = sorted(ecols, key=lambda x: int(x[1:]))
        return ecols_sorted

    # If neither pattern matches, abort with a helpful error
    raise RuntimeError(
        f"Could not locate embedding columns in {EMB_PATH}. "
        f"Got columns: {list(df.columns)[:20]} ..."
    )


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine distance between two 1D vectors.

    Cosine distance is defined as:
        d_cos(x, y) = 1 - cos_sim(x, y)
                    = 1 - (x · y) / (||x|| * ||y||)

    We L2-normalise both vectors to make the measure robust to scale.

    Parameters
    ----------
    a : numpy.ndarray
        1D embedding vector.
    b : numpy.ndarray
        1D embedding vector.

    Returns
    -------
    float
        Cosine distance in [0, 2] (with typical values closer to 0 for
        similar embeddings).
    """
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return 1.0 - float(np.dot(a, b))


def _compute_centroids(X: np.ndarray, y: np.ndarray) -> dict[int, np.ndarray]:
    """
    Compute a centroid embedding for each identity class.

    This is used for an additional open-set check: an instance is only
    accepted as belonging to a predicted class if it lies sufficiently
    close to that class centroid in cosine space.

    Parameters
    ----------
    X : numpy.ndarray
        2D array of embeddings, shape (n_samples, dim).
    y : numpy.ndarray
        1D array of identity labels aligned with the rows of X. Labels
        are expected to be numeric IDs.

    Returns
    -------
    dict[int, numpy.ndarray]
        Mapping from integer label to a normalised centroid embedding.
    """
    # L2-normalise all embeddings (FaceNet-style geometry)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    centroids: dict[int, np.ndarray] = {}

    for lab in np.unique(y):
        # Mean of all normalised embeddings for this identity
        c = Xn[y == lab].mean(axis=0)
        # Re-normalise centroid for stability in cosine calculations
        c /= (np.linalg.norm(c) + 1e-8)
        centroids[int(lab)] = c

    return centroids


def _load_embeddings() -> pd.DataFrame:
    """
    Load the embeddings DataFrame, enforcing appropriate dtypes.

    Expects `embeddings_with_unknown.parquet` as produced by
    scripts/create_unknown_split.py.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing embedding dimensions, 'identity', 'split',
        and 'image_id' columns.

    Raises
    ------
    SystemExit
        If the embeddings file is missing.
    """
    if not EMB_PATH.exists():
        raise SystemExit(f"Embeddings file not found: {EMB_PATH}")

    df = pd.read_parquet(EMB_PATH)

    # Normalise identity to integer if present
    if "identity" in df.columns:
        df["identity"] = df["identity"].astype(int)

    # Ensure 'split' is string-based (e.g., 'enroll', 'test', 'unknown')
    if "split" in df.columns:
        df["split"] = df["split"].astype(str)

    return df


def _load_models():
    """
    Load the trained classifier and label encoder.

    The classifier is assumed to be a scikit-learn KNN (or compatible
    estimator with predict_proba), and the label encoder maps between
    integer indices and original identity labels.

    Returns
    -------
    clf : object
        Trained classifier loaded via joblib.
    le : object
        Label encoder loaded via joblib.

    Raises
    ------
    SystemExit
        If either the classifier or label encoder file is missing.
    """
    clf_path = MODELS_DIR / "classifier.joblib"
    le_path = MODELS_DIR / "label_encoder.joblib"

    if not clf_path.exists() or not le_path.exists():
        raise SystemExit(f"Missing classifier/label encoder in {MODELS_DIR}")

    clf = load(clf_path)
    le = load(le_path)
    return clf, le


# ----------------------------------------------------------------------
# Core evaluation logic
# ----------------------------------------------------------------------
def evaluate_open_set(
    tau: float,
    dist_threshold: float | None = None
) -> dict[str, float]:
    """
    Evaluate open-set behaviour at a single operating point.

    The workflow is:
        1. Load embeddings and split into 'enroll' and 'test'.
        2. Build class centroids from the enrolment embeddings.
        3. Obtain closed-set predictions from the trained classifier
           (Top-1 prediction with probability scores).
        4. For each test sample:
            - Determine predicted class and class probability.
            - Compute cosine distance to the corresponding class centroid.
        5. Apply the open-set decision rule:
               accept if (prob >= tau) and (distance <= dist_threshold
                                          if provided)
           Otherwise, the sample is treated as UNKNOWN / rejected.
        6. Compute:
            - Closed-set Top-1 accuracy (ignoring open-set rejection).
            - Open-set Top-1 accuracy restricted to accepted samples.
            - Overall rejection rate.
            - Proportion of closed-set misclassifications that are
              successfully rejected.
            - Macro precision/recall/F1 on accepted predictions only.

    Parameters
    ----------
    tau : float
        Probability threshold for acceptance. Predictions with a
        maximum class probability below tau are automatically rejected.
    dist_threshold : float or None, optional
        Cosine distance threshold to centroid. If provided, predictions
        are only accepted if their distance to the predicted class
        centroid is <= dist_threshold. If None, only the probability
        constraint is applied.

    Returns
    -------
    dict[str, float]
        Dictionary of key metrics:
            - 'closed_top1'
            - 'open_top1'
            - 'reject_rate'
            - 'miscls_rejected'
            - 'accepted_macro_precision'
            - 'accepted_macro_recall'
            - 'accepted_macro_f1'
    """
    # Load embeddings and identify embedding columns
    df = _load_embeddings()
    emb_cols = _find_embedding_columns(df)

    # Basic split handling: we use 'enroll' for centroids and classifier,
    # and 'test' as the evaluation set. The 'unknown' split is used during
    # embedding creation and threshold tuning, but this script focuses on
    # enrolled vs non-enrolled behaviour via thresholds.
    if "split" not in df.columns:
        raise SystemExit("Column 'split' not found in embeddings parquet.")

    enroll = df[df["split"] == "enroll"].copy()
    test = df[df["split"] == "test"].copy()
    if enroll.empty or test.empty:
        raise SystemExit("Expected non-empty 'enroll' and 'test' splits.")

    # Feature matrices and identity labels
    X_enroll = enroll[emb_cols].to_numpy(dtype=np.float32)
    y_enroll = enroll["identity"].to_numpy(dtype=int)
    X_test = test[emb_cols].to_numpy(dtype=np.float32)
    y_test = test["identity"].to_numpy(dtype=int)

    # Load classifier and label encoder
    clf, le = _load_models()

    # Compute centroids using enrolment embeddings
    centroids = _compute_centroids(X_enroll, y_enroll)  # keys are raw identity IDs

    # Closed-set predictions (probabilities over known classes)
    probs = clf.predict_proba(X_test)  # shape (N, C)
    pred_idx = probs.argmax(axis=1)    # index of most likely class
    pred_lab = le.inverse_transform(pred_idx)  # back to identity IDs
    pred_prob = probs[np.arange(len(probs)), pred_idx]

    # Cosine distances from test embeddings to predicted centroids
    dists = np.zeros(len(X_test), dtype=np.float32)

    # Work in cosine geometry, so normalise test embeddings
    Xn_test = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-8)
    for i, lab in enumerate(pred_lab):
        c = centroids.get(int(lab), None)
        if c is None:
            # If we somehow lack a centroid for the predicted class,
            # treat the distance as very large so it will be rejected
            dists[i] = 9.9
        else:
            dists[i] = _cosine_distance(Xn_test[i], c)

    # ------------------------------------------------------------------
    # Open-set decision rule
    # ------------------------------------------------------------------
    # First, check probability
    accept_prob = pred_prob >= float(tau)

    # Then optionally require closeness to centroid
    if dist_threshold is not None:
        accept_dist = dists <= float(dist_threshold)
        accepts = accept_prob & accept_dist
    else:
        accepts = accept_prob

    # Closed-set correctness ignores the open-set gate
    closed_correct = (pred_lab == y_test)

    # Open-set correctness: a prediction counts as correct only if it is
    # both correct AND accepted by the open-set rule.
    open_correct = closed_correct & accepts

    top1_acc_closed = accuracy_score(y_test, pred_lab)  # baseline closed-set
    top1_acc_open = open_correct.mean() if len(open_correct) else 0.0
    reject_rate = (~accepts).mean()

    # Among all rejected instances, measure what fraction were already
    # misclassified under closed-set evaluation. This reflects how much
    # of the classifier's error is successfully "caught" by the open-set
    # rejection rule.
    miscls_rejected = ((~closed_correct) & (~accepts)).mean()

    # Macro precision/recall/F1 restricted to accepted examples only
    if accepts.any():
        y_true_acc = y_test[accepts].astype(int)
        y_pred_acc = pred_lab[accepts].astype(int)

        prec = precision_score(
            y_true_acc, y_pred_acc, average="macro", zero_division=0
        )
        rec = recall_score(
            y_true_acc, y_pred_acc, average="macro", zero_division=0
        )
        f1 = f1_score(
            y_true_acc, y_pred_acc, average="macro", zero_division=0
        )
    else:
        prec = rec = f1 = 0.0

    # Console report for quick inspection
    print(f"\n=== Open-Set Evaluation ===")
    print(f"Enroll N={len(X_enroll)}  Test N={len(X_test)}  Classes={len(np.unique(y_enroll))}")
    print(f"Closed-set Top-1 Acc: {top1_acc_closed:.4f}")
    if dist_threshold is None:
        print(f"Open-set (prob≥{tau:.2f}) Top-1 Acc on accepted: {top1_acc_open:.4f}")
    else:
        print(
            f"Open-set (prob≥{tau:.2f} & dist≤{dist_threshold:.3f}) "
            f"Top-1 Acc on accepted: {top1_acc_open:.4f}"
        )
    print(f"Reject rate: {reject_rate:.3f}")
    print(f"Misclassifications rejected as UNKNOWN: {miscls_rejected:.3f}")
    print(f"Accepted macro P/R/F1: {prec:.4f} / {rec:.4f} / {f1:.4f}")

    return {
        "closed_top1": float(top1_acc_closed),
        "open_top1": float(top1_acc_open),
        "reject_rate": float(reject_rate),
        "miscls_rejected": float(miscls_rejected),
        "accepted_macro_precision": float(prec),
        "accepted_macro_recall": float(rec),
        "accepted_macro_f1": float(f1),
    }


def _parse_range(spec: str) -> np.ndarray:
    """
    Parse a 'start:stop:step' string into a numpy array.

    The range is treated as inclusive on the stop value, within normal
    floating-point tolerance. This is convenient for threshold sweeps.

    Example
    -------
    '0.50:0.95:0.05' -> array([0.50, 0.55, ..., 0.95])

    Parameters
    ----------
    spec : str
        String specification of the form 'start:stop:step'.

    Returns
    -------
    numpy.ndarray
        1D array of threshold values.
    """
    a, b, c = spec.split(":")
    start, stop, step = float(a), float(b), float(c)
    # inclusive stop using integer rounding
    n = int(round((stop - start) / step)) + 1
    return start + np.arange(n) * step


def sweep(prob_spec: str | None,
          dist_spec: str | None,
          report_path: str | None):
    """
    Sweep probability and/or distance thresholds and log metrics.

    This function iterates over a grid of (tau, dist_threshold) values
    and collects open-set metrics for each configuration.

    Parameters
    ----------
    prob_spec : str or None
        Range specification 'start:stop:step' for probability threshold
        tau. If None, a single default value of 0.80 is used.
    dist_spec : str or None
        Range specification 'start:stop:step' for distance threshold.
        If None, no distance constraint is applied (probability only).
    report_path : str or None
        Path to a CSV file where sweep results will be stored.

    Raises
    ------
    SystemExit
        If report_path is not provided when a sweep is requested.
    """
    if report_path is None:
        raise SystemExit("Please provide --report when using --sweep_prob/--sweep_dist.")

    results: list[dict[str, float]] = []

    prob_vals = _parse_range(prob_spec) if prob_spec else [0.80]
    dist_vals = _parse_range(dist_spec) if dist_spec else [None]

    for tau in prob_vals:
        for d in dist_vals:
            metrics = evaluate_open_set(
                tau=float(tau),
                dist_threshold=(float(d) if d is not None else None),
            )
            row: dict[str, float | None] = {
                "tau": float(tau),
                "dist": (None if d is None else float(d)),
            }
            row.update(metrics)
            results.append(row)

    out = pd.DataFrame(results)
    out.to_csv(report_path, index=False)
    print(f"\nSaved sweep to {report_path}")


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
def main():
    """
    Command-line interface for open-set evaluation.

    By default, this evaluates a single operating point using the given
    probability and optional distance thresholds. If any sweep arguments
    are provided, it instead explores a grid of thresholds and writes
    the results to the specified report CSV.
    """
    ap = argparse.ArgumentParser(description="Open-set evaluation on saved embeddings.")
    ap.add_argument(
        "--tau",
        type=float,
        default=0.80,
        help="Probability threshold for acceptance.",
    )
    ap.add_argument(
        "--dist",
        type=float,
        default=None,
        help="Cosine distance-to-centroid threshold.",
    )
    ap.add_argument(
        "--sweep_prob",
        type=str,
        default=None,
        help="Range 'start:stop:step' to sweep probability threshold.",
    )
    ap.add_argument(
        "--sweep_dist",
        type=str,
        default=None,
        help="Range 'start:stop:step' to sweep distance threshold.",
    )
    ap.add_argument(
        "--report",
        type=str,
        default=None,
        help="CSV path to save sweep results (required if any sweep is used).",
    )
    args = ap.parse_args()

    # If any sweep range is provided, run the grid search over thresholds
    if args.sweep_prob or args.sweep_dist:
        sweep(args.sweep_prob, args.sweep_dist, args.report)
    else:
        # Single operating point evaluation
        evaluate_open_set(tau=args.tau, dist_threshold=args.dist)


if __name__ == "__main__":
    main()
