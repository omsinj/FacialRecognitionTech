# scripts/eval_far_frr.py
"""
eval_far_frr.py

Purpose
-------
Compute False Acceptance Rate (FAR), False Rejection Rate (FRR), and the
Equal Error Rate (EER) for stored face embeddings, using either cosine
similarity or Euclidean distance (or both).

This script:
    - Loads embeddings for a chosen split (e.g. "test").
    - Constructs genuine and impostor pairs.
    - Computes similarity / distance scores for those pairs.
    - Sweeps a range of thresholds to produce FAR/FRR curves.
    - Finds the threshold that minimises |FAR - FRR| (EER point).
    - Optionally saves curves (CSV/XLSX) and ROC/DET plots.

Typical usage
-------------
From the project root:

    python scripts/eval_far_frr.py --split test --metric both \
        --out_csv data/meta/evaluation/far_frr_curves.csv \
        --roc data/meta/evaluation/far_frr_roc.png \
        --det data/meta/evaluation/far_frr_det.png

This script is used to:
    - Quantify verification behaviour for cosine vs Euclidean metrics.
    - Select operating thresholds for the open-set access control decision.
"""

import os
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# matplotlib is optional so that the script can run headless on servers
try:
    import matplotlib.pyplot as plt
    _HAVE_PLT = True
except Exception:
    _HAVE_PLT = False

# Project config (falls back to defaults if fr_utils.config is not importable)
try:
    from fr_utils.config import META_DIR, SEED
except Exception:
    META_DIR = "data/meta"
    SEED = 42


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def _numeric_embedding_columns(df: pd.DataFrame):
    """
    Identify the embedding columns in the embeddings table.

    Parquet sometimes preserves the numeric embedding indices as strings
    ("0","1",...) and sometimes as ints (0,1,...). This helper:
        - Finds all columns whose labels are purely digits.
        - Sorts them numerically.
        - Returns them in the same type (str/int) as they appear in `df`.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame loaded from embeddings.parquet.

    Returns
    -------
    list
        Sorted list of column labels that correspond to embedding dimensions.

    Raises
    ------
    ValueError
        If no numeric embedding columns are found.
    """
    emb_cols = []
    for c in df.columns:
        # Only keep columns that look like numeric indices
        if str(c).isdigit():
            emb_cols.append(int(c))
    emb_cols = sorted(emb_cols)
    if not emb_cols:
        raise ValueError(
            "No numeric embedding columns found in embeddings.parquet. "
            "Expected columns like '0','1',...,'511'."
        )

    # Convert back to original type (string vs int) as present in the DataFrame
    use_str = isinstance(df.columns[0], str)
    return [str(i) if use_str else i for i in emb_cols]


def load_embeddings(parquet_path: str, split: str):
    """
    Load embeddings and metadata for a specific split.

    Parameters
    ----------
    parquet_path : str
        Path to embeddings.parquet (produced by build_embeddings.py).
    split : str
        Split to select, e.g. 'test' or 'enroll'.

    Returns
    -------
    X : numpy.ndarray
        2D array of embeddings with shape (n_samples, embedding_dim).
    y : numpy.ndarray
        1D array of identity labels (as strings), length n_samples.
    file_ids : numpy.ndarray
        1D array of image_id values corresponding to each embedding.

    Raises
    ------
    ValueError
        If required columns are missing or the split has no rows.
    """
    df = pd.read_parquet(parquet_path)

    # Ensure expected metadata columns are present
    for needed in ["identity", "split", "image_id"]:
        if needed not in df.columns:
            raise ValueError(
                f"Column '{needed}' not found in {parquet_path} "
                f"(have {list(df.columns)[:10]} ...)"
            )

    emb_cols = _numeric_embedding_columns(df)

    # Restrict to requested split
    sub = df[df["split"] == split].copy()
    if sub.empty:
        raise ValueError(f"No rows for split='{split}' in {parquet_path}.")

    # Standardise identity type to str
    sub["identity"] = sub["identity"].astype(str)

    X = sub[emb_cols].to_numpy(dtype=np.float32)
    y = sub["identity"].to_numpy()
    file_ids = sub["image_id"].to_numpy()

    return X, y, file_ids


def make_pairs(labels: np.ndarray,
               max_genuine_per_id: int | None = None,
               rng: np.random.Generator | None = None):
    """
    Build genuine and impostor index pairs for FAR/FRR analysis.

    Genuine pairs:
        - Two indices with the same identity label.
        - Optionally capped per identity for speed (max_genuine_per_id).

    Impostor pairs:
        - Two indices with different labels.
        - Sampled randomly so that the count is similar to the genuine set.

    Parameters
    ----------
    labels : numpy.ndarray
        1D array of identities (strings).
    max_genuine_per_id : int or None, optional
        Maximum number of genuine pairs to keep per identity. If None, all
        combinations are used. Default is None.
    rng : numpy.random.Generator or None, optional
        Random number generator instance for reproducibility. If None, a new
        default_rng is constructed using SEED.

    Returns
    -------
    genuine : list[tuple[int, int]]
        List of index pairs (i, j) where labels[i] == labels[j].
    impostor : list[tuple[int, int]]
        List of index pairs (i, j) where labels[i] != labels[j].

    Raises
    ------
    ValueError
        If no genuine pairs can be formed (e.g. too few samples per identity).
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    from collections import defaultdict

    # Bucket indices per identity
    buckets = defaultdict(list)
    for idx, lab in enumerate(labels):
        buckets[lab].append(idx)

    # Enumerate all genuine pairs, optionally sub-sampling for speed
    genuine = []
    for lab, idxs in buckets.items():
        if len(idxs) < 2:
            continue
        idxs = list(idxs)
        local_pairs = []
        # All combinations i < j
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                local_pairs.append((idxs[a], idxs[b]))
        if max_genuine_per_id is not None and len(local_pairs) > max_genuine_per_id:
            # Randomly down-sample genuine pairs for this identity
            local_pairs = rng.choice(local_pairs, size=max_genuine_per_id, replace=False).tolist()
        genuine.extend(local_pairs)

    n_genuine = len(genuine)
    if n_genuine == 0:
        raise ValueError("Could not form any genuine pairs — too few images per identity?")

    # Build impostor pairs by random sampling, targeting a similar count to genuine
    n = len(labels)
    impostor = set()
    needed = n_genuine
    tries = 0
    while len(impostor) < needed and tries < 50 * needed:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            tries += 1
            continue
        if labels[i] == labels[j]:
            tries += 1
            continue
        a, b = (i, j) if i < j else (j, i)
        impostor.add((a, b))
        tries += 1
    impostor = list(impostor)

    if len(impostor) < n_genuine:
        print(f"[warn] Only built {len(impostor)} impostor pairs vs {n_genuine} genuine.")

    return genuine, impostor


def cosine_scores(X: np.ndarray, pairs: list[tuple[int, int]]):
    """
    Compute cosine similarity scores for a set of index pairs.

    Cosine similarity is defined as:
        cos_sim(x, y) = (x · y) / (||x|| * ||y||)

    We L2-normalise each embedding row beforehand so that the dot product
    directly yields cosine similarity.

    Parameters
    ----------
    X : numpy.ndarray
        2D array of embeddings (n_samples, dim).
    pairs : list[tuple[int, int]]
        List of index pairs (i, j).

    Returns
    -------
    scores : numpy.ndarray
        1D array of cosine similarity scores for each pair.
    """
    # Normalise each row to unit length
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    scores = np.fromiter(
        (float((Xn[i] @ Xn[j])) for i, j in pairs),
        dtype=np.float32,
        count=len(pairs)
    )
    return scores


def l2_scores(X: np.ndarray, pairs: list[tuple[int, int]]):
    """
    Compute L2 (Euclidean) distances for a set of index pairs.

    Euclidean distance is defined as:
        d(x, y) = ||x - y||_2

    Parameters
    ----------
    X : numpy.ndarray
        2D array of embeddings (n_samples, dim).
    pairs : list[tuple[int, int]]
        List of index pairs (i, j).

    Returns
    -------
    scores : numpy.ndarray
        1D array of Euclidean distances for each pair.
    """
    scores = np.fromiter(
        (float(np.linalg.norm(X[i] - X[j])) for i, j in pairs),
        dtype=np.float32,
        count=len(pairs),
    )
    return scores


def sweep_thresholds(genuine_scores: np.ndarray,
                     impostor_scores: np.ndarray,
                     metric: str,
                     steps: int = 1000):
    """
    Sweep thresholds to compute FAR/FRR curves and locate the EER.

    Thresholding convention:
        - cosine:    higher = more similar -> match if score >= tau
        - euclidean: lower  = more similar -> match if dist  <= tau

    For each threshold `tau`, we compute:
        FAR = P(impostor accepted)
        FRR = P(genuine rejected)

    The Equal Error Rate (EER) is approximated by the threshold where
    |FAR - FRR| is minimised.

    Parameters
    ----------
    genuine_scores : numpy.ndarray
        1D array of similarity/distance values for genuine pairs.
    impostor_scores : numpy.ndarray
        1D array of similarity/distance values for impostor pairs.
    metric : {"cosine", "euclidean"}
        Which metric convention to use for interpreting scores.
    steps : int, optional
        Number of thresholds to sweep between the min and max of scores.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns ['tau', 'FAR', 'FRR'].
    eer : float
        Equal Error Rate at the best threshold.
    best_tau : float
        Threshold value that minimises |FAR - FRR|.
    """
    assert metric in ("cosine", "euclidean")

    g = np.asarray(genuine_scores, dtype=np.float64)
    imp = np.asarray(impostor_scores, dtype=np.float64)

    # Sweep thresholds over the combined score range
    lo, hi = float(min(g.min(), imp.min())), float(max(g.max(), imp.max()))
    taus = np.linspace(lo, hi, steps)

    if metric == "cosine":
        # FAR: impostor accepted (imp >= tau)
        # FRR: genuine rejected (g < tau)
        FAR = (imp[:, None] >= taus[None, :]).mean(axis=0)
        FRR = (g[:, None] <  taus[None, :]).mean(axis=0)
    else:
        # For Euclidean, smaller is more similar:
        # FAR: impostor accepted (imp <= tau)
        # FRR: genuine rejected (g > tau)
        FAR = (imp[:, None] <= taus[None, :]).mean(axis=0)
        FRR = (g[:, None] >  taus[None, :]).mean(axis=0)

    df = pd.DataFrame({"tau": taus, "FAR": FAR, "FRR": FRR})

    # Identify approximate EER: point where |FAR - FRR| is minimal
    idx = np.argmin(np.abs(FAR - FRR))
    eer = float((FAR[idx] + FRR[idx]) / 2.0)
    best_tau = float(taus[idx])

    return df, eer, best_tau


def maybe_save_curves(df_cos: pd.DataFrame | None,
                      df_l2: pd.DataFrame | None,
                      out_csv: str | None,
                      roc_path: str | None,
                      det_path: str | None):
    """
    Optionally persist the FAR/FRR curves and visualisations.

    Parameters
    ----------
    df_cos : pandas.DataFrame or None
        FAR/FRR sweep results for cosine similarity (or None if not computed).
    df_l2 : pandas.DataFrame or None
        FAR/FRR sweep results for Euclidean distance (or None if not computed).
    out_csv : str or None
        If provided, path to save concatenated curves (CSV or XLSX).
    roc_path : str or None
        If provided, path to save ROC-like plot (PNG).
    det_path : str or None
        If provided, path to save DET-like plot (PNG).

    Notes
    -----
    - CSV/XLSX output contains an additional 'metric' column to distinguish
      cosine vs Euclidean results.
    - ROC plots: TPR (1-FRR) vs FAR.
    - DET plots: FRR vs FAR in log-log space, as a simple approximation to
      traditional DET visualisations.
    """
    # Save tabular curves if requested
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        if out_csv.lower().endswith((".xlsx", ".xls")):
            # Excel writer: one sheet per metric where available
            with pd.ExcelWriter(out_csv) as writer:
                if df_cos is not None:
                    df_cos.assign(metric="cosine").to_excel(
                        writer, sheet_name="cosine", index=False
                    )
                if df_l2 is not None:
                    df_l2.assign(metric="euclidean").to_excel(
                        writer, sheet_name="euclidean", index=False
                    )
        else:
            # CSV: concatenate both metrics into a single file
            frames = []
            if df_cos is not None:
                frames.append(df_cos.assign(metric="cosine"))
            if df_l2 is not None:
                frames.append(df_l2.assign(metric="euclidean"))
            pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False)

    # If matplotlib is not available, skip plotting
    if not _HAVE_PLT:
        return

    def _plot_common(ax, df, title, is_det=False):
        """
        Internal helper to draw ROC/DET-style curves for a given DataFrame.
        """
        if df is None:
            return
        FAR, FRR = df["FAR"].to_numpy(), df["FRR"].to_numpy()

        if is_det:
            # Simple log-log representation of FRR vs FAR to approximate a DET
            ax.plot(FAR + 1e-6, FRR + 1e-6)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("FAR")
            ax.set_ylabel("FRR")
        else:
            # ROC-like: TPR vs FAR ; TPR = 1 - FRR
            ax.plot(FAR, 1 - FRR)
            ax.set_xlabel("FAR (FPR)")
            ax.set_ylabel("TPR (1 - FRR)")

        ax.grid(True, alpha=0.3)
        ax.set_title(title)

    # ROC plot (cosine and optionally Euclidean on the same axes)
    if roc_path:
        os.makedirs(os.path.dirname(roc_path), exist_ok=True)
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        _plot_common(ax, df_cos, "Cosine ROC")
        if df_l2 is not None:
            ax.plot(df_l2["FAR"], 1 - df_l2["FRR"])
            ax.legend(["cosine", "euclidean"])
        fig.tight_layout()
        fig.savefig(roc_path, dpi=200)
        plt.close(fig)

    # DET-like plot (log-log FRR vs FAR)
    if det_path:
        os.makedirs(os.path.dirname(det_path), exist_ok=True)
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        _plot_common(ax, df_cos, "Cosine DET", is_det=True)
        if df_l2 is not None:
            ax.plot(df_l2["FAR"] + 1e-6, df_l2["FRR"] + 1e-6)
            ax.legend(["cosine", "euclidean"])
        fig.tight_layout()
        fig.savefig(det_path, dpi=200)
        plt.close(fig)


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    """
    Command-line interface for FAR/FRR and EER evaluation.

    Reads embeddings from a parquet file, constructs genuine/impostor pairs,
    computes cosine / Euclidean scores, sweeps thresholds, and prints/saves
    the resulting metrics and curves.
    """
    ap = argparse.ArgumentParser(
        description="Compute FAR/FRR and EER from stored embeddings."
    )
    ap.add_argument(
        "--parquet", default=os.path.join(META_DIR, "embeddings.parquet"),
        help="Path to embeddings.parquet"
    )
    ap.add_argument(
        "--split", default="test", choices=["test", "enroll"],
        help="Which split to use for pairs"
    )
    ap.add_argument(
        "--max_genuine_per_id", type=int, default=None,
        help="Cap genuine pairs per identity (for speed)."
    )
    ap.add_argument(
        "--steps", type=int, default=1000,
        help="Number of thresholds to sweep."
    )
    ap.add_argument(
        "--metric", choices=["both", "cosine", "euclidean"], default="both",
        help="Which metric(s) to evaluate."
    )
    ap.add_argument(
        "--out_csv", default=None,
        help="Optional path to save sweep curves (csv/xlsx)."
    )
    ap.add_argument(
        "--roc", default=None,
        help="Optional path to save ROC plot (png)."
    )
    ap.add_argument(
        "--det", default=None,
        help="Optional path to save DET plot (png)."
    )
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)

    # 1) Load embeddings for the chosen split
    X, y, file_ids = load_embeddings(args.parquet, split=args.split)

    # 2) Build genuine and impostor pairs
    genuine, impostor = make_pairs(
        y,
        max_genuine_per_id=args.max_genuine_per_id,
        rng=rng
    )

    # 3) Compute similarity/distance scores
    run_cos = args.metric in ("both", "cosine")
    run_l2 = args.metric in ("both", "euclidean")

    df_cos = df_l2 = None
    if run_cos:
        g_cos = cosine_scores(X, genuine)
        i_cos = cosine_scores(X, impostor)
        df_cos, eer_cos, tau_cos = sweep_thresholds(
            g_cos, i_cos, metric="cosine", steps=args.steps
        )

    if run_l2:
        g_l2 = l2_scores(X, genuine)
        i_l2 = l2_scores(X, impostor)
        df_l2, eer_l2, tau_l2 = sweep_thresholds(
            g_l2, i_l2, metric="euclidean", steps=args.steps
        )

    # 4) Persist curves and plots if requested
    maybe_save_curves(df_cos, df_l2, args.out_csv, args.roc, args.det)

    # 5) Console report
    print("\n=== SETTINGS ===")
    print(f"Split: {args.split}")
    print(f"Embeddings: {args.parquet}")
    print(f"Genuine pairs: {len(genuine)} | Impostor pairs: {len(impostor)}")
    print(f"Threshold steps: {args.steps}")

    print("\n=== RESULTS ===")
    if run_cos:
        print(f"Cosine Similarity: Best Threshold = {tau_cos:.3f}, EER = {eer_cos:.3f}")
    if run_l2:
        print(f"Euclidean Distance: Best Threshold = {tau_l2:.3f}, EER = {eer_l2:.3f}")

    # Optional: show the head of each curve table for quick inspection
    if df_cos is not None:
        head = df_cos.head(3)
        print("\n[cosine curve head]")
        print(head.to_string(index=False))
    if df_l2 is not None:
        head = df_l2.head(3)
        print("\n[euclidean curve head]")
        print(head.to_string(index=False))


if __name__ == "__main__":
    # Keep legacy NumPy RNG consistent where used, and then run CLI.
    np.random.seed(SEED)
    main()
