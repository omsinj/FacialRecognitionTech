import io
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import cv2
from joblib import load

# --- Project imports
from fr_utils.config import (
    MODELS_DIR, META_DIR, ALIGNED_DIR, THRESHOLD as DEFAULT_VERIFY_THR
)
from fr_utils.facenet_backend import embed_image

# =========================
# Streamlit page config & global styling
# =========================
st.set_page_config(
    page_title="FR Access Control • Demo",
    page_icon="🔐",
    layout="wide",
)

st.markdown("""
<style>
/* App-wide */
:root {
  --bg: #0f172a;
  --card: #111827;
  --text: #e5e7eb;
  --muted: #9ca3af;
  --accent: #22d3ee;
  --danger: #ef4444;
  --success: #34d399;
  --warning: #f59e0b;
}
html, body, [class*="css"]  {
  color: var(--text) !important;
  background-color: var(--bg) !important;
}
section.main > div { max-width: 1400px; }
.block-container { padding-top: 1.6rem; }

.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 16px;
  padding: 18px 18px 16px 18px;
  box-shadow: 0 8px 30px rgba(0,0,0,0.2);
}
.metric {
  display: flex; align-items: center; gap: 12px;
  padding: 10px 12px; border-radius: 12px;
  background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06);
}
.pill { padding: 2px 10px; border-radius: 999px; font-weight: 600; display: inline-block; }
.pill-success { background: rgba(52,211,153,0.18); color: #a7f3d0; border: 1px solid rgba(52,211,153,0.35); }
.pill-danger  { background: rgba(239,68,68,0.18);  color: #fecaca; border: 1px solid rgba(239,68,68,0.35);  }
.pill-warn    { background: rgba(245,158,11,0.18); color: #fde68a; border: 1px solid rgba(245,158,11,0.35); }
.small { color: var(--muted); font-size: 0.9rem; }
hr { border-color: rgba(255,255,255,0.08); }
</style>
""", unsafe_allow_html=True)

# =========================
# Session-level stats
# =========================
# Track how many decisions have been made in this Streamlit session.
if "stats" not in st.session_state:
    st.session_state.stats = {"n": 0, "accepted": 0, "rejected": 0}

# =========================
# Utilities: loading models & data
# =========================

@st.cache_resource(show_spinner=False)
def load_models():
    """
    Load the trained classifier and label encoder from the models directory.

    This is cached as a resource because these objects are relatively heavy to
    load and do not change while the app is running.
    """
    clf = load(Path(MODELS_DIR) / "classifier.joblib")
    le = load(Path(MODELS_DIR) / "label_encoder.joblib")
    return clf, le

@st.cache_data(show_spinner=False)
def load_enroll_embeddings():
    """
    Load the enrolment split embeddings and compute per-identity centroids.

    Returns:
        X_enroll (np.ndarray): enrolment embeddings, shape (N, D)
        y_enroll (np.ndarray): enrolment identities as strings, shape (N,)
        centroids (pd.DataFrame): per-identity centroid vectors indexed by identity
    """
    pqt = Path(META_DIR) / "embeddings.parquet"
    if not pqt.exists():
        st.error(f"Missing embeddings parquet at {pqt}. Run build_embeddings.py first.")
        st.stop()

    df = pd.read_parquet(pqt)

    # Feature columns are numeric-like (0..511), stored as int or str in Parquet
    feat_cols = [
        c for c in df.columns
        if (isinstance(c, int) or (isinstance(c, str) and c.isdigit()))
    ]

    enroll = df[df["split"] == "enroll"].copy()

    # Compute class centroids directly in embedding space
    centroids = (
        enroll
        .groupby("identity")[feat_cols]
        .mean()
        .astype(np.float32)
    )

    X_enroll = enroll[feat_cols].values.astype(np.float32)
    y_enroll = enroll["identity"].astype(str).values

    return X_enroll, y_enroll, centroids

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Args:
        a, b (np.ndarray): vectors of identical dimensionality.

    Returns:
        float: cosine similarity in [-1, 1], where 1 indicates identical direction.
    """
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))

def nearest_centroid_distance(emb: np.ndarray, centroids_df: pd.DataFrame, label: str) -> float:
    """
    Compute cosine distance (1 - cosine similarity) between an embedding and
    the centroid of a given identity.

    Args:
        emb (np.ndarray): embedding vector to evaluate.
        centroids_df (pd.DataFrame): per-identity centroids indexed by 'identity'.
        label (str): identity label whose centroid should be used.

    Returns:
        float: cosine distance; larger values indicate the embedding is further
               from the class centroid. Returns +inf if the label has no centroid.
    """
    if label not in centroids_df.index:
        return float("inf")
    c = centroids_df.loc[label].values.astype(np.float32)
    return 1.0 - cosine_similarity(emb, c)

def decide_open_set(probs, pred_label, emb_vec, centroids_df, prob_thr: float, dist_thr: float):
    """
    Apply an open-set gate over closed-set predictions.

    Decision rule:
        Accept the prediction if BOTH:
            - max class probability >= prob_thr
            - cosine distance to predicted-class centroid <= dist_thr

    Args:
        probs (np.ndarray): class probability vector from the classifier.
        pred_label (str): predicted identity label.
        emb_vec (np.ndarray): embedding of the probe image.
        centroids_df (pd.DataFrame): per-identity centroids.
        prob_thr (float): minimum acceptable class probability.
        dist_thr (float): maximum acceptable centroid distance.

    Returns:
        (accepted: bool, top_prob: float, dist: float)
    """
    top_prob = float(np.max(probs))
    dist = nearest_centroid_distance(emb_vec, centroids_df, pred_label)
    accepted = (top_prob >= prob_thr) and (dist <= dist_thr)
    return accepted, top_prob, dist

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """
    Convert a PIL image into an OpenCV BGR image suitable for the embedding backend.

    Args:
        pil_img (PIL.Image): input image in arbitrary mode.

    Returns:
        np.ndarray: BGR uint8 image.
    """
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def render_decision_card(accepted: bool, pred_label: str, prob: float, dist: float,
                         prob_thr: float, dist_thr: float):
    """
    Render a compact decision summary (GRANTED / DENIED) with the main numeric
    quantities shown.

    Args:
        accepted (bool): final open-set decision.
        pred_label (str): identity label predicted by the classifier.
        prob (float): maximum class probability for the prediction.
        dist (float): cosine distance to the predicted-class centroid.
        prob_thr (float): active probability threshold.
        dist_thr (float): active distance threshold.
    """
    status = "GRANTED" if accepted else "DENIED"
    pill_cls = "pill-success" if accepted else "pill-danger"
    prob_status = "✓" if prob >= prob_thr else "✗"
    dist_status = "✓" if dist <= dist_thr else "✗"
    
    st.markdown(f"""
    <div class="card">
      <div class="metric">
        <span class="pill {pill_cls}">{status}</span>
        <div>
          <div><strong>Predicted ID:</strong> {pred_label}</div>
          <div class="small">
            <span>Prob: {prob:.3f} {prob_status} {prob_thr:.2f}</span> • 
            <span>Dist: {dist:.3f} {dist_status} {dist_thr:.3f}</span>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# Sidebar: thresholds & paths
# =========================
with st.sidebar:
    st.markdown("### 🔧 Decision thresholds")
    
    # Quick summary of current settings
    st.markdown("""
    <div style="background: rgba(34, 211, 238, 0.1); padding: 10px; border-radius: 8px; margin-bottom: 15px;">
    <small>Current settings:</small><br>
    <strong>Prob ≥ {prob_thr:.2f}</strong> • <strong>Dist ≤ {dist_thr:.3f}</strong>
    </div>
    """, unsafe_allow_html=True)

    # Threshold presets for quick switching between operating regimes.
    preset = st.radio(
        "Presets",
        ["Balanced", "Convenience", "Strict"],
        horizontal=True,
    )

    # Visual feedback for preset selection
    preset_colors = {
        "Convenience": "🟢",
        "Balanced": "🟡", 
        "Strict": "🔴"
    }
    st.caption(f"{preset_colors[preset]} {preset} mode selected")

    if preset == "Convenience":
        # lenient: higher TPR, higher FAR
        prob_default, dist_default = 0.60, 1.10
    elif preset == "Strict":
        # strict: low FAR, more rejections
        prob_default, dist_default = 0.90, 0.75
    else:
        # balanced: close to your reported EER-inspired settings
        prob_default, dist_default = 0.80, 0.90

    prob_thr = st.slider("Min. class probability", 0.00, 1.00, prob_default, 0.01,
                        help="Minimum confidence required. Higher = more strict (fewer false acceptances).")
    dist_thr = st.slider("Max. centroid cosine distance", 0.000, 2.000, dist_default, 0.001,
                        help="Maximum allowed distance from person's centroid. Lower = more strict (face must be very similar).")
    st.caption("Open-set rule: Accept iff `prob ≥ prob_thr` **and** `dist ≤ dist_thr`.")

    st.markdown("---")
    st.markdown("### 📁 Data")
    st.write(f"**Models**: `{MODELS_DIR}`")
    st.write(f"**Embeddings**: `{Path(META_DIR) / 'embeddings.parquet'}`")
    st.write(f"**Aligned images**: `{ALIGNED_DIR}`")
    
    # Quick tips expander
    with st.expander("🎯 Quick tips"):
        st.markdown("""
        - **Upload**: Drag & drop or click to browse
        - **Webcam**: Grant camera permission when prompted
        - **Gallery**: Pre-loaded test images
        - Adjust sliders to see how thresholds affect decisions
        - Check "Why this decision?" for detailed breakdown
        """)

# =========================
# Header and global info
# =========================
st.markdown("## 🔐 Facial Recognition Access Control — Prototype Demo")

clf, le = load_models()
_, _, centroids = load_enroll_embeddings()

colA, colB = st.columns([1, 1])

with colA:
    st.markdown(
        "Use **Upload**, **Webcam**, or the **Gallery** to test the prototype. "
        "Decisions are made using a trained classifier on FaceNet embeddings, "
        "combined with an **open-set** gate over identity centroids."
    )

with colB:
    stats = st.session_state.stats
    accept_rate = (stats["accepted"] / stats["n"]) if stats["n"] > 0 else 0.0
    
    # Add reset button
    reset_col1, reset_col2 = st.columns([3, 1])
    with reset_col1:
        st.markdown(
            f"<div class='card'>"
            f"<strong>Models loaded:</strong> {clf.__class__.__name__} • {len(centroids)} identities<br/>"
            f"<span class='small'>Session: {stats['n']} attempts • "
            f"{stats['accepted']} granted • {stats['rejected']} denied • "
            f"Accept rate: {accept_rate:.1%}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with reset_col2:
        if st.button("↺", help="Reset session stats"):
            st.session_state.stats = {"n": 0, "accepted": 0, "rejected": 0}
            st.rerun()

with st.expander("ℹ️ About this demo & data handling"):
    st.markdown(
        "- All inference is performed locally in this prototype; no images are sent to external services.\n"
        "- Uploaded or captured images are processed in-memory by Streamlit for the current session.\n"
        "- The classifier and centroids are trained on a subset of the CelebA dataset only.\n"
        "- This interface is intended for research and teaching purposes rather than production deployment."
    )

st.markdown("---")

# =========================
# Tabs: Upload / Webcam / Gallery / Metrics
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["⬆️ Upload", "📷 Webcam", "🖼️ Gallery", "📊 Metrics"])

def process_and_decide(pil_img: Image.Image):
    """
    End-to-end processing for a single image:
        - Convert PIL -> BGR
        - Embed via FaceNet
        - Predict closed-set identity
        - Apply open-set decision rule
        - Update session statistics
        - Render decision card + explanation panel
    """
    # Show processing spinner
    with st.spinner("Processing image..."):
        # 1) Convert & embed
        bgr = pil_to_bgr(pil_img)
        try:
            emb = embed_image(bgr).astype(np.float32)  # shape (512,)
        except Exception as e:
            st.error(f"Face not detected or could not be aligned: {e}")
            return

        # 2) Closed-set prediction with classifier
        probs = clf.predict_proba(emb.reshape(1, -1))[0]
        top_idx = int(np.argmax(probs))
        pred_label = str(le.inverse_transform([top_idx])[0])

        # 3) Open-set decision
        accepted, top_prob, dist = decide_open_set(
            probs, pred_label, emb, centroids, prob_thr, dist_thr
        )

        # 4) Update session stats
        st.session_state.stats["n"] += 1
        if accepted:
            st.session_state.stats["accepted"] += 1
        else:
            st.session_state.stats["rejected"] += 1

        # 5) Render core decision
        render_decision_card(accepted, pred_label, top_prob, dist, prob_thr, dist_thr)

        # 6) Explain the decision: top-k classes and distances
        top_indices = np.argsort(probs)[::-1][:5]
        top_labels = le.inverse_transform(top_indices)
        top_probs = probs[top_indices]

        dist_rows = []
        for lab in top_labels[:3]:
            d = nearest_centroid_distance(emb, centroids, str(lab))
            dist_rows.append({"identity": str(lab), "cosine_dist": d})
        dist_df = pd.DataFrame(dist_rows)

        with st.expander("🔎 Why this decision?", expanded=True):
            # Add visual bar chart for probabilities
            prob_chart_data = pd.DataFrame({
                'Identity': [str(l) for l in top_labels],
                'Probability': top_probs
            })
            
            st.bar_chart(prob_chart_data.set_index('Identity')['Probability'])
            
            st.write("**Detailed breakdown:**")
            
            st.write("**Top predictions (closed-set classifier)**")
            pred_df = pd.DataFrame({
                "identity": [str(l) for l in top_labels],
                "probability": top_probs,
            })
            st.dataframe(
                pred_df.style.format({"probability": "{:.3f}"}),
                use_container_width=True,
            )

            st.write("**Distances to centroids (lower = closer)**")
            st.dataframe(
                dist_df.style.format({"cosine_dist": "{:.3f}"}),
                use_container_width=True,
            )

            st.caption(
                "The open-set rule accepts the prediction only when the classifier is confident "
                "(high probability) and the embedding lies close to the predicted class centroid "
                "in embedding space."
            )

        # 7) Show the input image
        st.image(pil_img, caption="Input image", use_column_width=True)

# -------------------------
# Tab 1: File upload
# -------------------------
with tab1:
    st.subheader("Upload a face image")
    file = st.file_uploader("JPEG/PNG", type=["jpg", "jpeg", "png"])
    if file:
        img = Image.open(io.BytesIO(file.read()))
        process_and_decide(img)

# -------------------------
# Tab 2: Webcam capture
# -------------------------
with tab2:
    st.subheader("Capture from webcam")
    cam = st.camera_input("Take a photo")
    if cam is not None:
        img = Image.open(cam)
        process_and_decide(img)

# -------------------------
# Tab 3: Identity gallery (from aligned/ + subset.csv)
# -------------------------
with tab3:
    st.subheader("Quick test from aligned gallery")

    subset_csv = Path(META_DIR) / "subset.csv"
    if subset_csv.exists():
        df = pd.read_csv(subset_csv)
        df["identity"] = df["identity"].astype(str)

        sample_ids = sorted(df["identity"].unique())
        pick_id = st.selectbox("Choose identity", sample_ids)

        df_id = df[(df["identity"] == pick_id) & (df["split"] == "test")]

        if not df_id.empty:
            # Show up to 4 thumbnails for context
            st.markdown("**Sample images for this identity (test split)**")
            thumbs = df_id["image_id"].tolist()[:4]
            cols = st.columns(len(thumbs))
            for c, fname in zip(cols, thumbs):
                img_path = Path(ALIGNED_DIR) / fname
                if img_path.exists():
                    with c:
                        st.image(str(img_path), caption=fname, use_column_width=True)
            
            # Add quick test button
            if st.button(f"🔍 Test random image for {pick_id}", type="secondary"):
                fname = random.choice(df_id["image_id"].tolist())
                img_path = Path(ALIGNED_DIR) / fname
                if img_path.exists():
                    st.markdown(f"**Evaluating random image:** `{fname}`")
                    img = Image.open(str(img_path))
                    process_and_decide(img)
            else:
                # Original random selection logic
                fname = random.choice(df_id["image_id"].tolist())
                img_path = Path(ALIGNED_DIR) / fname
                if img_path.exists():
                    st.markdown("---")
                    st.markdown(f"**Evaluating:** `{fname}`")
                    img = Image.open(str(img_path))
                    process_and_decide(img)
                else:
                    st.warning("Selected image missing in aligned/. Run preprocess_align.py again.")
        else:
            st.info("No test images for this identity.")
    else:
        st.info("subset.csv not found. Run your subset/build steps first.")

# -------------------------
# Tab 4: Offline metrics overview
# -------------------------
with tab4:
    st.subheader("Offline evaluation summary")

    # FAR/FRR curves from eval_far_frr.py (cosine)
    far_path = Path(META_DIR) / "evaluation" / "far_frr_cosine.csv"
    if far_path.exists():
        st.markdown("**FAR/FRR curve (cosine similarity)**")
        far_df = pd.read_csv(far_path)
        # Plot FAR and FRR over threshold to show the trade-off
        st.line_chart(far_df[["FAR", "FRR"]])
        st.caption("Trade-off between False Acceptance Rate (FAR) and False Rejection Rate (FRR).")
    else:
        st.info("far_frr_cosine.csv not found. Run eval_far_frr.py to generate FAR/FRR curves.")

    # Confusion matrix visualization
    conf_matrix_path = Path(META_DIR) / "evaluation" / "confusion_matrix.csv"
    if conf_matrix_path.exists():
        st.markdown("---")
        st.markdown("**Confusion Matrix**")
        conf_matrix = pd.read_csv(conf_matrix_path, index_col=0)
        
        # Create a heatmap using matplotlib
        try:
            import matplotlib.pyplot as plt
            
            # Limit to top 20 identities for readability
            top_identities = conf_matrix.index[:20]
            conf_matrix_top = conf_matrix.loc[top_identities, top_identities]
            
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(conf_matrix_top.values, cmap='Blues')
            
            # Add labels
            ax.set_xticks(np.arange(len(conf_matrix_top.columns)))
            ax.set_yticks(np.arange(len(conf_matrix_top.index)))
            ax.set_xticklabels(conf_matrix_top.columns, rotation=45, ha="right")
            ax.set_yticklabels(conf_matrix_top.index)
            
            # Add text annotations for high values only
            for i in range(len(conf_matrix_top.index)):
                for j in range(len(conf_matrix_top.columns)):
                    if conf_matrix_top.iloc[i, j] > 0.1:  # Only show significant values
                        text = ax.text(j, i, f"{conf_matrix_top.iloc[i, j]:.1f}",
                                      ha="center", va="center", 
                                      color="black" if conf_matrix_top.iloc[i, j] < 0.5 else "white")
            
            ax.set_title(f"Confusion Matrix (Top {len(top_identities)} Identities)")
            plt.tight_layout()
            st.pyplot(fig)
        except ImportError:
            st.info("Matplotlib not available for confusion matrix visualization")
    else:
        st.info("confusion_matrix.csv not found. Run evaluation scripts to generate.")

    st.markdown("---")

    # Robustness results from robustness_eval.py
    robust_path = Path(META_DIR) / "robustness_results.csv"
    if robust_path.exists():
        st.markdown("**Robustness: accuracy under perturbations**")
        rob = pd.read_csv(robust_path)

        # Aggregate mean accuracy per condition (e.g. baseline, sunglasses, blur, etc.)
        agg = (
            rob.groupby("condition", as_index=False)
               .agg(acc=("acc", "mean"))
               .sort_values("acc", ascending=False)
        )
        agg = agg.set_index("condition")

        st.bar_chart(agg[["acc"]])
        st.caption("Average closed-set accuracy for each robustness condition (baseline, sunglasses, blur, etc.).")

        # Show a small table snapshot
        with st.expander("Show raw robustness summary"):
            st.dataframe(rob.head(30), use_container_width=True)
    else:
        st.info("robustness_results.csv not found. Run robustness_eval.py to generate robustness metrics.")