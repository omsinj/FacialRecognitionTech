import os
from dataclasses import dataclass

# Project roots
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
ALIGNED_DIR = os.path.join(DATA_DIR, "aligned")
META_DIR = os.path.join(DATA_DIR, "meta")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ensure dirs exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(ALIGNED_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Reproducibility & model settings
SEED = 42
IMAGE_SIZE = (160, 160)   # FaceNet expected input
MARGIN = 10               # extra pixels around detected face crop
THRESHOLD = 0.90          # default cosine similarity threshold for 1:1 (verify_pair)

# Column schema for embeddings parquet
# We’ll store the embedding columns as f0..f511 (strings) to avoid mixed-name issues.
EMB_COLS = [f"f{i}" for i in range(512)]
ID_COL = "identity"
SPLIT_COL = "split"
FILE_COL = "image_id"
