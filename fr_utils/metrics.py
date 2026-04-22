import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_curve, auc

def compute_basic_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

def compute_far_frr(y_true_ids, y_scores, pos_label):
    """Compute FAR/FRR at a threshold for a 1-vs-rest scenario.
    - y_true_ids: array of true labels (ids)
    - y_scores: array of similarity/scores for the positive class
    - pos_label: the label considered 'genuine'
    Returns FAR, FRR across thresholds plus ROC AUC.
    """
    y_true_bin = (y_true_ids == pos_label).astype(int)
    fpr, tpr, thresholds = roc_curve(y_true_bin, y_scores)
    roc_auc = auc(fpr, tpr)
    # FAR is FPR; FRR is 1-TPR
    far = fpr
    frr = 1 - tpr
    return far, frr, thresholds, roc_auc
