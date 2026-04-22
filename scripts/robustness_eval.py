# scripts/robustness_eval.py
"""
robustness_eval.py

Purpose
-------
Evaluate how the closed-set classifier behaves when test images are
subjected to controlled, synthetic perturbations that approximate
real-world capture conditions (occlusion, shadows, blur, noise, rotation,
brightness shifts).

This script:
    • loads the trained classifier + label encoder
    • reads the test split from embeddings.parquet to get (image_id, identity)
    • reloads the *aligned* face crops from disk
    • applies a range of perturbations per image (e.g. sunglasses, shadows,
      rotation, blur, noise, brightness)
    • re-embeds each perturbed image using embed_image (FaceNet backend)
    • runs the classifier on the new embeddings
    • computes accuracy, macro precision/recall/F1, and Top-5 accuracy
      for each perturbation type and parameter setting
    • optionally generates matplotlib plots summarising degradation

Outputs
-------
    • CSV: robustness_results.csv in data/meta/ (or path specified via --out_csv)
    • Optional PNG plots (if --plots_dir is provided):
         - robustness_accuracy_by_condition.png
         - robustness_degradation_vs_baseline.png

This is an *evaluation* tool, not part of the runtime access-control path.
"""

import os
import argparse
import math
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import load
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Project utils and paths
from fr_utils.facenet_backend import embed_image
from fr_utils.config import META_DIR, ALIGNED_DIR, MODELS_DIR, SEED


# -----------------------
# Image perturbations
# -----------------------

def put_sunglasses(img, opacity=1.0):
    """
    Apply a crude sunglasses-like occlusion over the upper face region.

    Parameters
    ----------
    img : np.ndarray
        Input RGB image of shape (H, W, 3), assumed to be a 160×160 aligned face.
    opacity : float
        Opacity of the black rectangle, in [0.0, 1.0]. Higher = stronger occlusion.

    Returns
    -------
    np.ndarray
        RGB image with a dark rectangle over the top portion (periocular area).
    """
    h, w = img.shape[:2]
    # Define vertical bounds of the occlusion band (roughly eye region)
    y1, y2 = int(h * 0.25), int(h * 0.45)
    overlay = img.copy()
    cv2.rectangle(overlay, (int(w * 0.15), y1), (int(w * 0.85), y2), (0, 0, 0), -1)
    # Blend occlusion with original using given opacity
    return cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0)


def add_shadow(img, strength=0.6, angle_deg=45):
    """
    Apply a slanted multiplicative gradient to simulate cast shadow.

    Parameters
    ----------
    img : np.ndarray
        Input RGB image (H, W, 3).
    strength : float
        Shadow strength in [0, 1]. Larger values yield darker shadows.
    angle_deg : float
        Angle of the gradient in degrees, controlling shadow direction.

    Returns
    -------
    np.ndarray
        RGB image darkened by a directional gradient mask.
    """
    h, w = img.shape[:2]
    Y, X = np.mgrid[0:h, 0:w]
    angle = math.radians(angle_deg)

    # Normalised coordinates centred around (0,0)
    Xc = (X - w / 2) / (w / 2)
    Yc = (Y - h / 2) / (h / 2)

    # Project onto shadow direction
    grad = (np.cos(angle) * Xc + np.sin(angle) * Yc)
    # Normalise to [0,1]
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
    mask = 1.0 - strength * grad
    mask = np.clip(mask, 0.0, 1.0)

    out = (img.astype(np.float32) * mask[..., None]).clip(0, 255).astype(np.uint8)
    return out


def rotate(img, degrees=15):
    """
    Rotate the image by a given number of degrees around its centre.

    Uses border reflection to avoid black borders, approximating off-angle capture.

    Parameters
    ----------
    img : np.ndarray
        RGB input image.
    degrees : float
        Rotation angle; positive = counter-clockwise.

    Returns
    -------
    np.ndarray
        Rotated RGB image of same size.
    """
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def gaussian_blur(img, ksize=5):
    """
    Apply Gaussian blur to approximate motion blur or low-resolution capture.

    Parameters
    ----------
    img : np.ndarray
        RGB input image.
    ksize : int
        Kernel size (pixels). Will be forced to an odd integer >= 3.

    Returns
    -------
    np.ndarray
        Blurred RGB image.
    """
    k = max(3, int(ksize) | 1)  # ensure odd >= 3
    return cv2.GaussianBlur(img, (k, k), 0)


def gaussian_noise(img, sigma=10):
    """
    Add zero-mean Gaussian noise to simulate sensor noise or compression artefacts.

    Parameters
    ----------
    img : np.ndarray
        RGB input image.
    sigma : float
        Standard deviation of the Gaussian noise (pixel intensity units).

    Returns
    -------
    np.ndarray
        Noisy RGB image.
    """
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def brightness(img, delta=-40):
    """
    Apply a global brightness shift, simulating under/over-exposure.

    Parameters
    ----------
    img : np.ndarray
        RGB input image.
    delta : int
        Additive offset applied to all channels. Negative darkens, positive brightens.

    Returns
    -------
    np.ndarray
        Brightness-adjusted RGB image.
    """
    out = img.astype(np.int16) + int(delta)
    return np.clip(out, 0, 255).astype(np.uint8)


# -----------------------
# Data helpers
# -----------------------

def load_test_manifest():
    """
    Load (image_id, identity) pairs from the test split.

    We rely on embeddings.parquet only for split metadata. The actual
    pixel data is reloaded from data/aligned via image_id.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ["image_id", "identity"] for split=='test'.
        Identity is cast to string to keep sklearn label handling consistent.
    """
    df = pd.read_parquet(os.path.join(META_DIR, "embeddings.parquet"))
    manifest = df[df["split"] == "test"][["image_id", "identity"]].copy()
    # Avoid sklearn "Mix of label input types" errors
    manifest["identity"] = manifest["identity"].astype(str)
    return manifest


def safe_imread_rgb(folder, image_id):
    """
    Load an aligned face image from disk as RGB.

    Parameters
    ----------
    folder : str or Path
        Directory containing aligned images (ALIGNED_DIR).
    image_id : str
        File name of the aligned face (e.g. '000001.jpg').

    Returns
    -------
    np.ndarray | None
        RGB image if load succeeds, else None.
    """
    bgr = cv2.imread(os.path.join(folder, image_id))
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# -----------------------
# Eval core
# -----------------------

def run_eval(clf,
             le,
             image_ids,
             labels,
             transform_name,
             transform_fn,
             limit=None):
    """
    Run a closed-set evaluation under a specific perturbation pipeline.

    For each image:
        1. Load aligned RGB image from ALIGNED_DIR.
        2. Apply transform_fn (e.g. put_sunglasses / add_shadow / rotate / ...).
        3. Convert to BGR and embed using embed_image.
        4. Classify embedding using the trained classifier.
        5. Compare predicted labels against ground truth to compute metrics.

    Parameters
    ----------
    clf : sklearn-like classifier
        Trained classifier with predict_proba or predict.
    le : LabelEncoder
        Maps string identity labels to integer class indices.
    image_ids : list[str]
        List of image file names (aligned faces).
    labels : list[str]
        List of ground-truth identity labels (string form).
    transform_name : str
        Name used for progress bar and reporting (e.g. "sunglasses(op=1.0)").
    transform_fn : Callable[[np.ndarray], np.ndarray]
        Function mapping RGB -> RGB, applying the chosen perturbation.
    limit : int | None
        Optional cap on number of images to process (for speed).

    Returns
    -------
    dict
        Dictionary summarising performance:
            {
                "n": number_of_evaluated_samples,
                "acc": top-1 accuracy,
                "top5": top-5 accuracy,
                "prec": macro-precision,
                "rec": macro-recall,
                "f1": macro-F1
            }
        If no samples were evaluated, returns zeros for all metrics.
    """
    y_true = []
    y_pred = []
    top5_hits = []

    items = list(zip(image_ids, labels))
    if limit is not None:
        items = items[:limit]

    for fname, lab in tqdm(items, desc=f"{transform_name}", ncols=80):
        # Load RGB aligned face
        rgb = safe_imread_rgb(ALIGNED_DIR, fname)
        if rgb is None:
            # Missing/invalid image: skip
            continue

        # Apply synthetic perturbation in RGB space
        rgb_t = transform_fn(rgb)

        # embed_image expects BGR input
        try:
            emb = embed_image(cv2.cvtColor(rgb_t, cv2.COLOR_RGB2BGR)).reshape(1, -1)
        except Exception:
            # If face detection/alignment fails post-perturbation,
            # treat as a miss by skipping any prediction.
            continue

        # Closed-set prediction
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(emb)[0]
            top1 = probs.argmax()
            top5 = np.argsort(probs)[-5:][::-1]
        else:
            # kNN should typically provide predict_proba=True, but we guard anyway.
            top1 = int(clf.predict(emb)[0])
            top5 = np.array([top1])

        # Encode ground-truth label to match classifier index space
        try:
            y_true_enc = int(le.transform([str(lab)])[0])
        except ValueError:
            # Identity not present in encoder (should not happen on test set)
            continue

        y_true.append(y_true_enc)
        y_pred.append(int(top1))
        top5_hits.append(1 if y_true_enc in top5 else 0)

    if not y_true:
        # No valid samples processed under this perturbation
        return {"n": 0, "acc": 0.0, "top5": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0}

    # Aggregate metrics
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return {
        "n": len(y_true),
        "acc": acc,
        "top5": float(np.mean(top5_hits)),
        "prec": prec,
        "rec": rec,
        "f1": f1,
    }


# -----------------------
# Plotting (matplotlib, no seaborn)
# -----------------------

def make_plots(df, out_dir):
    """
    Generate simple bar plots summarising robustness results.

    Produces:
        • robustness_accuracy_by_condition.png
            - mean accuracy per perturbation condition

        • robustness_degradation_vs_baseline.png
            - accuracy drop relative to baseline

    Parameters
    ----------
    df : pandas.DataFrame
        Robustness results with columns including:
            ["condition", "param", "acc", ...]
    out_dir : str or Path
        Directory where PNG plots will be written.
    """
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # Aggregate per condition by averaging accuracy over all parameter values
    agg = (
        df.groupby("condition", as_index=False)
          .agg(acc=("acc", "mean"))
          .sort_values("acc", ascending=False)
    )

    # --- Plot 1: Accuracy by condition
    plt.figure(figsize=(10, 5))
    plt.bar(agg["condition"], agg["acc"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Robustness: Accuracy by Condition (mean over params)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "robustness_accuracy_by_condition.png"),
        dpi=220,
    )
    plt.close()

    # --- Plot 2: Degradation relative to baseline
    base = df[df["condition"] == "baseline"]["acc"].mean()
    df2 = agg.copy()
    df2["degradation"] = base - df2["acc"]

    plt.figure(figsize=(10, 5))
    plt.bar(df2["condition"], df2["degradation"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Δ Accuracy vs Baseline")
    plt.title("Degradation relative to Baseline")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "robustness_degradation_vs_baseline.png"),
        dpi=220,
    )
    plt.close()


def main():
    """
    CLI entry point.

    Typical usage
    -------------
    python -m scripts.robustness_eval --plots_dir data/meta --limit 300

    Flags allow you to:
        • limit the number of test images (for speed)
        • control parameter grids for each perturbation type
        • override the CSV output path
        • optionally generate summary plots
    """
    ap = argparse.ArgumentParser(description="Evaluate robustness of FR model under synthetic perturbations.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of test images for speed.")
    ap.add_argument("--rot", type=int, nargs="*", default=[10, 20, 30],
                    help="Rotation degrees to test (± each value).")
    ap.add_argument("--shadow", type=float, nargs="*", default=[0.3, 0.6],
                    help="Shadow strengths in [0,1].")
    ap.add_argument("--sunglasses", type=float, nargs="*", default=[1.0],
                    help="Opacity values for sunglasses occlusion.")
    ap.add_argument("--blur", type=int, nargs="*", default=[3, 7],
                    help="Gaussian blur kernel sizes.")
    ap.add_argument("--noise", type=float, nargs="*", default=[8, 15],
                    help="Gaussian noise sigmas.")
    ap.add_argument("--bright", type=int, nargs="*", default=[-30, -50, 30],
                    help="Brightness deltas.")
    ap.add_argument("--out_csv", type=str, default=None,
                    help="Override output CSV path (default: data/meta/robustness_results.csv).")
    ap.add_argument("--plots_dir", type=str, default=None,
                    help="If set, also save robustness plots into this directory.")
    args = ap.parse_args()

    np.random.seed(SEED)

    # Load classifier + label encoder (trained on enrolment embeddings)
    clf = load(os.path.join(MODELS_DIR, "classifier.joblib"))
    le  = load(os.path.join(MODELS_DIR, "label_encoder.joblib"))

    # Build test manifest of (image_id, identity) pairs
    manifest = load_test_manifest()
    image_ids = manifest["image_id"].tolist()
    labels    = manifest["identity"].tolist()

    results = []

    def add_result(name, stats, param=None):
        """
        Append a single (condition, param, metrics) row to the results list.

        Only non-empty evaluations (n > 0) are recorded.
        """
        if stats["n"] == 0:
            return
        row = {"condition": name, "param": "" if param is None else str(param)}
        row.update(stats)
        results.append(row)

    # --- Baseline (no perturbation)
    add_result(
        "baseline",
        run_eval(clf, le, image_ids, labels, "baseline", lambda x: x, args.limit),
    )

    # --- Sunglasses / eye occlusion
    for op in args.sunglasses:
        add_result(
            "sunglasses",
            run_eval(
                clf,
                le,
                image_ids,
                labels,
                f"sunglasses(op={op})",
                lambda x, o=op: put_sunglasses(x, opacity=o),
                args.limit,
            ),
            op,
        )

    # --- Shadows
    for s in args.shadow:
        add_result(
            "shadow",
            run_eval(
                clf,
                le,
                image_ids,
                labels,
                f"shadow(str={s})",
                lambda x, st=s: add_shadow(x, strength=st, angle_deg=45),
                args.limit,
            ),
            s,
        )

    # --- Rotation (± each degree value)
    for deg in args.rot:
        add_result(
            "rot+deg",
            run_eval(
                clf,
                le,
                image_ids,
                labels,
                f"rot(+{deg})",
                lambda x, d=deg: rotate(x, d),
                args.limit,
            ),
            f"+{deg}",
        )
        add_result(
            "rot-deg",
            run_eval(
                clf,
                le,
                image_ids,
                labels,
                f"rot(-{deg})",
                lambda x, d=deg: rotate(x, -d),
                args.limit,
            ),
            f"-{deg}",
        )

    # --- Gaussian blur
    for k in args.blur:
        add_result(
            "blur",
            run_eval(
                clf,
                le,
                image_ids,
                labels,
                f"blur({k})",
                lambda x, kk=k: gaussian_blur(x, kk),
                args.limit,
            ),
            k,
        )

    # --- Gaussian noise
    for sig in args.noise:
        add_result(
            "noise",
            run_eval(
                clf,
                le,
                image_ids,
                labels,
                f"noise({sig})",
                lambda x, s=sig: gaussian_noise(x, s),
                args.limit,
            ),
            sig,
        )

    # --- Brightness shifts
    for delta in args.bright:
        add_result(
            "brightness",
            run_eval(
                clf,
                le,
                image_ids,
                labels,
                f"bright({delta})",
                lambda x, d=delta: brightness(x, d),
                args.limit,
            ),
            delta,
        )

    # --- Persist results
    out_df = pd.DataFrame(results)
    os.makedirs(META_DIR, exist_ok=True)
    out_csv = args.out_csv or os.path.join(META_DIR, "robustness_results.csv")
    out_df.to_csv(out_csv, index=False)

    print("\nROBUSTNESS SUMMARY (first 20 rows):")
    print(out_df.head(20).to_string(index=False))
    print(f"\nSaved full results to {out_csv}")

    # --- Optional plots
    if args.plots_dir:
        make_plots(out_df, args.plots_dir)
        print(f"Saved plots to {args.plots_dir}")


if __name__ == "__main__":
    main()
