# scripts/train_classifier.py
"""
train_classifier.py

Purpose
-------
Train a closed-set identity classifier on FaceNet embeddings and derive
auxiliary artefacts used by the access-control prototype:

    • Supervised classifier (SVM or k-NN) trained on enrolment embeddings.
    • LabelEncoder mapping identity strings → integer class indices.
    • Per-class centroids in L2-normalised embedding space.
    • Simple, data-driven open-set thresholds:
         - prob_thr  : minimum class probability for acceptance
         - sim_thr   : minimum cosine similarity to predicted-class centroid

These artefacts are later consumed by:
    - scripts/evaluate.py           (closed-set metrics)
    - scripts/eval_open_set.py      (open-set analysis)
    - scripts/robustness_eval.py    (under perturbations)
    - app/prototype_access.py       (command-line access decision)
    - app/streamlit_app.py          (Streamlit demo UI)

Inputs
------
    • data/meta/embeddings.parquet
        - columns "0".."511" (embedding dims) plus:
            - identity
            - split  ∈ {"enroll","test"}
            - image_id

Outputs
-------
    • models/classifier.joblib      – trained sklearn classifier
    • models/label_encoder.joblib   – fitted LabelEncoder
    • models/centroids.npz          – per-class centroids (L2-normalised)
    • models/model_meta.json        – prob_thr and sim_thr thresholds
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from joblib import dump

from fr_utils.facenet_backend import cosine_similarity
from fr_utils.config import META_DIR, MODELS_DIR, SEED


def choose_features(df: pd.DataFrame):
    """
    Infer which columns correspond to embedding dimensions.

    Parquet may persist columns as ints (0..511) or strings ("0".."511").
    This helper:
        • selects columns that are either ints OR digit-only strings
        • normalises them to string form so downstream code has a stable schema

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame loaded from embeddings.parquet.

    Returns
    -------
    list[str]
        List of column names (as strings) representing embedding dimensions.
    """
    # parquet saved numeric columns as 0..511; coerce to str and pick digit-like
    feat_cols = [
        c for c in df.columns
        if (isinstance(c, int) or (isinstance(c, str) and c.isdigit()))
    ]
    feat_cols = [str(c) for c in feat_cols]
    return feat_cols


def compute_centroids(X: np.ndarray, y: np.ndarray, labels: np.ndarray):
    """
    Compute an L2-normalised centroid for each identity in embedding space.

    Steps:
        1. L2-normalise each embedding row to project into cosine space.
        2. For each label in `labels`, average the normalised vectors belonging
           to that identity.
        3. L2-normalise the resulting centroid again to ensure unit length.

    These centroids are used for:
        • analysis (e.g. robustness_eval)
        • open-set distance gating (cosine distance to predicted-class centroid)

    Parameters
    ----------
    X : np.ndarray
        Embedding matrix of shape (N, D) for enrolment samples.
    y : np.ndarray
        Identity labels for each row in X (string or numeric, but consistent).
    labels : np.ndarray
        Sorted list of unique label values (aligned with LabelEncoder classes_).

    Returns
    -------
    dict
        Mapping {label: centroid_vector}, with centroid vectors as float32.
    """
    # L2-normalize embeddings before centroiding (cosine space)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    centroids = {}
    for lab in labels:
        idx = np.where(y == lab)[0]
        c = Xn[idx].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-8)
        centroids[lab] = c.astype(np.float32)
    return centroids


def calibrate_thresholds(X_enroll,
                         y_enroll,
                         X_val,
                         y_val,
                         le,
                         clf,
                         centroids):
    """
    Estimate simple open-set acceptance thresholds from a validation split.

    Two thresholds are derived:

        • prob_thr
            - Only used if classifier supports predict_proba.
            - Take the predicted probability of the *predicted* class.
            - Restrict to examples where the prediction is correct.
            - Set prob_thr to the 5th percentile of those probabilities.
              The intuition: accept predictions that are at least as confident
              as the bottom 5% of correctly classified known users.

        • sim_thr
            - Cosine similarity between the normalised embedding and the
              centroid of the *predicted* class.
            - Again, collect similarities for correctly handled cases and
              take the 5th percentile as the minimum acceptable similarity.

    N.B. In this implementation, `X_test` is used as a proxy for validation
    (i.e. we are not holding out a separate dev set). For a larger study, you
    would typically split enrol/test/val explicitly.

    Parameters
    ----------
    X_enroll : np.ndarray
        Enrolment embeddings (unused directly here, but passed for extensibility).
    y_enroll : np.ndarray
        Enrolment labels (unused here, but kept for future refinements).
    X_val : np.ndarray
        Validation / test embeddings used to calibrate thresholds.
    y_val : np.ndarray
        Ground-truth labels for X_val (string form).
    le : LabelEncoder
        Fitted label encoder.
    clf : sklearn classifier
        Trained classifier with predict or predict_proba.
    centroids : dict
        Per-class centroids as returned by compute_centroids.

    Returns
    -------
    tuple[float, float]
        prob_thr, sim_thr
    """
    # ---- Probability-based threshold (if classifier supports predict_proba)
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_val)
        pred = probs.argmax(axis=1)
        # True labels encoded into classifier index space
        correct = (pred == le.transform(y_val))
        if correct.any():
            # Probabilities of the chosen class for correctly classified samples
            top_probs = probs.max(axis=1)[correct]
            # 5th percentile keeps ~95% of correct known users
            prob_thr = float(np.percentile(top_probs, 5))
        else:
            prob_thr = 0.0
    else:
        # KNN with weights='distance' has predict_proba; if not, we fall back gracefully.
        prob_thr = 0.0

    # ---- Cosine similarity threshold w.r.t. predicted-class centroids
    Xn = X_val / (np.linalg.norm(X_val, axis=1, keepdims=True) + 1e-8)

    # We can use predict() for labels in either case (with or without predict_proba)
    pred = clf.predict(X_val)
    pred_labels = le.inverse_transform(pred)

    sims = []
    for vec, lab in zip(Xn, pred_labels):
        c = centroids.get(lab)
        if c is None:
            # Should not happen if centroids are built from same label set
            continue
        sims.append(cosine_similarity(vec, c))

    if len(sims):
        # Again use 5th percentile as a conservative lower bound
        sim_thr = float(np.percentile(np.array(sims), 5))
    else:
        sim_thr = 0.0

    return prob_thr, sim_thr


def main(model_type: str):
    """
    Orchestrate classifier training and threshold calibration.

    Steps
    -----
    1. Load embeddings.parquet and infer embedding columns.
    2. Split into:
           - enrol: training set for classifier + centroid estimation
           - test : pseudo-validation split for summarising performance and
                    calibrating thresholds.
    3. Encode labels using LabelEncoder to obtain integer class indices.
    4. Instantiate classifier (linear SVM or distance-weighted k-NN).
    5. Train classifier on enrolment embeddings.
    6. Print closed-set classification report on test split.
    7. Compute per-class centroids in L2-normalised space and save them.
    8. Calibrate simple open-set thresholds (prob_thr, sim_thr) on test data.
    9. Persist classifier, label encoder, centroids, and thresholds to models/.

    Parameters
    ----------
    model_type : str
        Either "svm" or "knn", controlling the classifier family.
    """
    # Load full embeddings table
    df = pd.read_parquet(os.path.join(META_DIR, "embeddings.parquet"))
    feat_cols = choose_features(df)

    # Split into enrolment (train) and test (held-out for reporting/calibration)
    enroll = df[df["split"] == "enroll"]
    test   = df[df["split"] == "test"]

    X_enroll = enroll[feat_cols].values.astype(np.float32)
    y_enroll = enroll["identity"].astype(str).values

    X_test   = test[feat_cols].values.astype(np.float32)
    y_test   = test["identity"].astype(str).values

    # Map identity strings → integer class indices
    le = LabelEncoder()
    y_enroll_enc = le.fit_transform(y_enroll)
    y_test_enc   = le.transform(y_test)

    # Choose classifier family
    if model_type == "svm":
        # Linear SVM with probability=True so we can use predict_proba downstream
        clf = SVC(C=1.0, kernel="linear", probability=True, random_state=SEED)
    elif model_type == "knn":
        # K=3, distance-weighted k-NN is a reasonable baseline for FaceNet embeddings
        clf = KNeighborsClassifier(n_neighbors=3, weights="distance", n_jobs=-1)
    else:
        raise ValueError("--model must be 'svm' or 'knn'")

    # Train closed-set classifier on enrolment embeddings
    clf.fit(X_enroll, y_enroll_enc)

    # Closed-set report on test partition (sanity check and summary for dissertation)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test_enc, y_pred, target_names=le.classes_)
    print(report)

    # ---- Compute and persist centroids (L2-normalised)
    centroids = compute_centroids(X_enroll, y_enroll, le.classes_)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Store centroids in a simple label-aligned matrix, plus label names
    np.savez(
        os.path.join(MODELS_DIR, "centroids.npz"),
        labels=le.classes_,
        centroids=np.vstack([centroids[lab] for lab in le.classes_]),
    )

    # ---- Calibrate thresholds using test split as proxy for validation
    prob_thr, sim_thr = calibrate_thresholds(
        X_enroll,
        y_enroll,
        X_test,
        y_test,
        le,
        clf,
        centroids,
    )

    meta = {"prob_thr": prob_thr, "sim_thr": sim_thr}
    with open(os.path.join(MODELS_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---- Save classifier and label encoder
    dump(clf, os.path.join(MODELS_DIR, "classifier.joblib"))
    dump(le,  os.path.join(MODELS_DIR, "label_encoder.joblib"))

    print(f"Saved classifier + label encoder to {MODELS_DIR}")
    print(f"Auto-calibrated thresholds: prob_thr={prob_thr:.3f}, sim_thr={sim_thr:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train closed-set FR classifier and calibrate simple open-set thresholds.")
    ap.add_argument(
        "--model",
        choices=["svm", "knn"],
        default="knn",
        help="Classifier type to train. 'knn' is the default used in the prototype.",
    )
    args = ap.parse_args()
    main(args.model)
