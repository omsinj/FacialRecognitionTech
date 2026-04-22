# scripts/evaluate.py
"""
evaluate.py

Purpose
-------
Compute closed-set classification performance for the FaceNet + KNN
pipeline using stored embeddings.

This script:
    - Loads the test split from embeddings_with_unknown.parquet.
    - Loads the trained classifier and label encoder from disk.
    - Computes:
        * Top-1 accuracy
        * Top-5 accuracy
        * Macro-averaged precision, recall, and F1-score
        * Average per-sample latency for prediction
    - Optionally generates and saves confusion matrix plots
      (both raw and normalised).

This provides the main closed-set baseline reported in the dissertation.
"""

import os
import time
import sys

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import matplotlib.pyplot as plt

from fr_utils.config import META_DIR, MODELS_DIR


def topk_accuracy(probs: np.ndarray, y_true: np.ndarray, k: int = 5) -> float:
    """
    Compute Top-k accuracy given class probability predictions.

    Parameters
    ----------
    probs : np.ndarray
        Array of shape (N, C) with class probabilities for N samples and C classes.
    y_true : np.ndarray
        True labels encoded as integer class indices (same encoding as in probs).
    k : int, optional
        The 'k' in Top-k (default is 5).

    Returns
    -------
    float
        Fraction of samples where the true class appears in the top-k
        most probable predictions.
    """
    # argsort in descending order and take first k indices per row
    topk = np.argsort(-probs, axis=1)[:, :k]
    # Check, for each sample, whether the true label is one of the top-k predictions
    return np.mean([y_true[i] in topk[i] for i in range(len(y_true))])


def save_confusion_matrix(
    cm: np.ndarray,
    class_names: np.ndarray,
    filename: str,
    normalize: bool = False,
) -> None:
    """
    Save a cleaner confusion matrix plot for the 50-identity closed-set evaluation.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix of shape (C, C), where C is the number of classes.
    class_names : np.ndarray
        Array of class labels corresponding to rows/columns of cm.
    filename : str
        Name of the output PNG file (saved under META_DIR/evaluation/).
    normalize : bool, optional
        If True, normalise rows to sum to 1 to show relative error patterns.
        If False, show raw counts.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    # To avoid clutter with 50 classes, only show every 5th identity index
    n_classes = len(class_names)
    tick_positions = np.arange(0, n_classes, 5)
    tick_labels = [str(i + 1) for i in tick_positions]  # 1-based indices

    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)

    ax.set_xlabel("Predicted identity index")
    ax.set_ylabel("True identity index")

    title = "Confusion matrix (normalised)" if normalize else "Confusion matrix"
    ax.set_title(title, fontsize=11)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    out_path = os.path.join(META_DIR, "evaluation")
    os.makedirs(out_path, exist_ok=True)
    full_path = os.path.join(out_path, filename)
    plt.savefig(full_path, dpi=300)
    plt.close(fig)
    print(f"Saved confusion matrix to {full_path}")


def main() -> None:
    """
    Entry point for closed-set evaluation.

    Steps
    -----
    1. Optionally parse '--plot_confusion' flag from sys.argv.
    2. Load embeddings_with_unknown.parquet from META_DIR and
       select the 'test' split for closed-set evaluation.
    3. Extract feature columns (embedding dimensions) and labels.
    4. Load the trained classifier and label encoder from MODELS_DIR.
    5. Time the prediction step to estimate average per-sample latency.
    6. Compute:
           - Top-1 accuracy
           - Top-5 accuracy
           - Macro precision, recall, F1
           - Average per-sample latency
    7. Optionally compute and save confusion matrices (raw + normalised).

    Notes
    -----
    - This evaluation uses the 'test' split only and ignores any
      'unknown' rows in the file.
    - The classifier is evaluated in a purely closed-set fashion;
      open-set logic is handled separately in eval_open_set.py.
    """
    # Command-line toggle to control whether confusion matrices are plotted/saved
    plot_confusion = "--plot_confusion" in sys.argv

    # Load embeddings + metadata (includes enroll/test/unknown splits)
    df = pd.read_parquet(os.path.join(META_DIR, "embeddings_with_unknown.parquet"))

    # Identify embedding feature columns: either int or digit-like strings
    feat_cols = [
        c
        for c in df.columns
        if isinstance(c, int) or (isinstance(c, str) and c.isdigit())
    ]

    # Restrict evaluation to the test split only
    X_test = df[df["split"] == "test"][feat_cols].to_numpy(dtype=np.float32)
    y_test = df[df["split"] == "test"]["identity"].astype(str).to_numpy()

    # Load classifier and label encoder
    clf = load(os.path.join(MODELS_DIR, "classifier.joblib"))
    le = load(os.path.join(MODELS_DIR, "label_encoder.joblib"))

    # Encode string labels into integer indices matching classifier outputs
    y_test_enc = le.transform(y_test)

    # Measure prediction latency per sample
    t0 = time.time()
    if hasattr(clf, "predict_proba"):
        # KNN from scikit-learn supports predict_proba
        probs = clf.predict_proba(X_test)
    else:
        # Fallback for classifiers without probability estimates:
        #  - get class predictions
        #  - build a "one-hot" probability matrix for compatibility
        pred = clf.predict(X_test)
        probs = np.zeros((len(pred), len(le.classes_)), dtype=np.float32)
        probs[np.arange(len(pred)), pred] = 1.0

    latency = (time.time() - t0) / max(1, len(X_test))

    # Closed-set predictions (class index with highest probability)
    y_pred = np.argmax(probs, axis=1)

    # Core metrics
    acc = accuracy_score(y_test_enc, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test_enc,
        y_pred,
        average="macro",
        zero_division=0,
    )
    top5 = topk_accuracy(probs, y_test_enc, k=5)

    # Console report
    print(f"Top-1 Accuracy: {acc:.4f}")
    print(f"Top-5 Accuracy: {top5:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall (macro): {rec:.4f}")
    print(f"F1 (macro): {f1:.4f}")
    print(f"Avg per-sample latency (s): {latency:.6f}")

    # Optionally compute and save confusion matrix plots
    if plot_confusion:
        cm = confusion_matrix(y_test_enc, y_pred)
        class_names = le.classes_
        save_confusion_matrix(cm, class_names, "confusion_matrix.png", normalize=False)
        save_confusion_matrix(
            cm,
            class_names,
            "confusion_matrix_normalised.png",
            normalize=True,
        )


if __name__ == "__main__":
    main()
