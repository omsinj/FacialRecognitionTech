# Facial Recognition for Privacy-Preserving Access Control
# Research Prototype Repository 

This repository accompanies the dissertation “Design and Evaluation of a Privacy-Preserving Facial Recognition System for Small-Scale Access Control.”
It contains the full implementation pipeline, evaluation suite, and analysis tools used to examine how a lightweight facial-recognition (FR) system can operate under strict privacy constraints, open-set verification requirements, and controlled robustness testing.

## The artefact employs:

MTCNN for face detection and alignment

FaceNet (512-D embeddings) for feature representation

KNN-based closed-set classification

Threshold-based open-set rejection mechanisms

Multi-dimensional evaluation framework aligned with ISO/IEC 19795 principles

## All processing occurs entirely on-device, and no new biometric data is collected.

1. Research Objectives

This repository operationalises the research aim:

To design, implement, and evaluate a privacy-preserving facial recognition prototype for access control, using a multi-criteria assessment of accuracy, open-set reliability, robustness, and data integrity.

## Specifically, the system supports investigations into:

Discriminative performance under closed-set identification

Open-set verification via FAR/FRR and EER analysis

Environmental robustness under controlled perturbations

Practical viability under constrained computational resources

Privacy-by-design system architecture and data-handling practices

All scripts reproduce the analyses reported in the dissertation.

2. Quick Start (Python 3.12)

Create and activate virtual environment
python3.12 -m venv .venv
source .venv/bin/activate       # macOS/Linux
.venv\Scripts\Activate.ps1      # Windows

## Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

## Prepare the dataset

CelebA images must be manually placed in data/raw/. Downloaded from https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

## Then:

python scripts/make_subset.py
python scripts/preprocess_align.py
python scripts/build_embeddings.py
python scripts/a_sanity_check.py

## Train and evaluate
python scripts/train_classifier.py
python scripts/build_centroids.py
python scripts/evaluate.py
python scripts/eval_far_frr.py
python scripts/eval_open_set.py
python scripts/robustness_eval.py
analyse_knn_confidence.py


## Run the prototype
python app/prototype_access.py --image path/to/face.jpg

or

streamlit run app/streamlit_app.py

3. Repository Structure
FacialRecognition/
├── app/                        # Access-control prototype (CLI + Streamlit)
├── data/                       # Raw, aligned images, embeddings, metadata
├── fr_utils/                   # FaceNet backend, distance metrics, open-set logic
├── models/                     # KNN model, label encoder, centroids, thresholds
├── scripts/                    # Full pipeline: preprocessing → evaluation
├── requirements.txt
└── README.md

4. Data Pipeline and Evaluation Framework

The repository implements the full experimental workflow used in the study:

4.1 Preprocessing

make_subset.py: selects 50 identities × 20 images from CelebA

preprocess_align.py: applies MTCNN to generate 160×160 aligned face crops

4.2 Embedding Generation

build_embeddings.py: extracts 512-D FaceNet embeddings

Stores results in a unified embeddings.parquet containing:

512 embedding columns

identity, split, image_id metadata

4.3 Model Training

train_classifier.py: trains KNN classifier for closed-set identification

build_centroids.py: computes per-identity centroids for distance-based verification

4.4 Closed-Set Evaluation

evaluate.py reports:

Top-1 / Top-5 accuracy

Macro precision, recall, F1

End-to-end pipeline latency

4.5 Open-Set Verification

eval_far_frr.py: FAR/FRR curves and EER for cosine and Euclidean metrics

eval_open_set.py: evaluates rejection behaviour with calibrated thresholds

4.6 Robustness Testing

robustness_eval.py evaluates:

Occlusion (sunglasses)

Illumination variation

Gaussian blur

Gaussian noise

Brightness shifts

Pose (rotation)

Outputs include performance degradation plots and condition-specific metrics.

4.7 Data Integrity Checks

audit_dataset.py: detects duplicates and leakage across splits

a_sanity_check.py: validates identity counts and split consistency

eval_shuffle_baseline.py: confirms that no spurious separability exists in the dataset

5. The Application Layer

Two interfaces demonstrate the prototype’s operational behaviour:

## CLI (prototype_access.py)

python app/prototype_access.py --image path/to/image.jpg

## Outputs include:

predicted identity

classifier probability

distance to identity centroid

final access decision (ACCEPT / DENIED (UNKNOWN))

## Streamlit Interface (streamlit_app.py)

Provides a visual, interactive demonstration of:

face detection and alignment

embedding similarity

decision thresholds

open-set rejection reasoning

6. Ethical, Legal, and Privacy Considerations

This project adheres strictly to privacy-by-design principles:

No new biometric data collected

All experimentation uses the publicly available CelebA dataset.

Local-only processing
No cloud services, uploads, or remote inference.

Minimal data retention
Only aligned crops and embeddings necessary for reproducibility are stored.

Open-set verification
Prevents deterministic misclassification of unknown individuals, reducing security risk.

No fairness auditing claims
CelebA lacks demographic labels; the system cannot be responsibly evaluated for demographic bias.

The system is therefore appropriate for academic experimentation but not authorised for real-world security deployment without further fairness, security, and proportionality assessments.

7. Reproducibility Notes

To support scholarly replication:

Random seeds are fixed where applicable

All intermediate files (aligned images, embeddings, centroids, evaluation CSVs) are preserved

FAR/FRR curves and robustness plots are generated deterministically

Model artefacts are stored in models/ with accompanying metadata JSON

8. Citation

If using this repository in academic work:

[Omar Jagana], "[Facial Recognition System for Access Control]," [Abertay University], [2025].


or cite the core components:

Schroff et al., FaceNet: A Unified Embedding for Face Recognition, CVPR 2015

Zhang et al., Joint Face Detection and Alignment via MTCNN, 2016

9. License (MIT)
MIT License

Copyright (c) 2025 [Omar Jagana]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS," WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



10. Acknowledgements

This project builds upon widely used open-source libraries, including TensorFlow/Keras, MTCNN, NumPy, Pandas, scikit-learn, and Streamlit.
Their contributions make reproducible research in FR possible.