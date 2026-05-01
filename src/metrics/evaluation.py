# evaluation.py
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

def compute_accuracy(y_true, y_pred):
    """
    Compute accuracy (proportion of correct predictions).
 
    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
 
    Returns:
        float: Accuracy score.
    """
    return accuracy_score(y_true, y_pred)
 
 
def compute_zeror(y_train):
    """
    Compute ZeroR baseline accuracy: always predicts the majority class
    of the training set. Used in the paper as a lower-bound reference.
 
    Args:
        y_train (np.ndarray): Training labels.
 
    Returns:
        float: ZeroR accuracy (= proportion of majority class in y_train).
    """
    values, counts = np.unique(y_train, return_counts=True)
    majority_class_count = counts.max()
    return majority_class_count / len(y_train)
 
 
def compute_f1(y_true, y_pred, average="weighted"):
    """
    Compute weighted F1-score (default), reflecting class distribution.
    Matches the aggregation strategy used in the paper.
 
    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        average (str): Averaging strategy ('weighted', 'macro', 'micro').
 
    Returns:
        float: F1-score.
    """
    return f1_score(y_true, y_pred, average=average, zero_division=0)
 
 
def compute_precision(y_true, y_pred, average="weighted"):
    """
    Compute weighted precision.
 
    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        average (str): Averaging strategy.
 
    Returns:
        float: Precision score.
    """
    return precision_score(y_true, y_pred, average=average, zero_division=0)
 
 
def compute_recall(y_true, y_pred, average="weighted"):
    """
    Compute weighted recall. Note: weighted recall == accuracy in
    multi-class single-label settings (as noted in the paper, Eq. 2).
 
    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        average (str): Averaging strategy.
 
    Returns:
        float: Recall score.
    """
    return recall_score(y_true, y_pred, average=average, zero_division=0)
 
 
def compute_normalized_improvement(accuracy, zeror):
    """
    Compute normalized improvement over ZeroR baseline (paper Eq. 4).
    NI = (Accuracy - ZeroR) / (1 - ZeroR)
 
    Args:
        accuracy (float): Model accuracy.
        zeror (float): ZeroR baseline accuracy.
 
    Returns:
        float: Normalized improvement.
    """
    if zeror >= 1.0:
        return 0.0
    return (accuracy - zeror) / (1 - zeror)
 
 
def compute_all_metrics(y_true, y_pred, y_train):
    """
    Compute all metrics reported in the paper's Table II:
    Accuracy, ZeroR, weighted F1, weighted Precision, weighted Recall,
    and Normalized Improvement.
 
    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        y_train (np.ndarray): Training labels (needed for ZeroR).
 
    Returns:
        dict: Dictionary with all metrics.
    """
    acc = compute_accuracy(y_true, y_pred)
    zeror = compute_zeror(y_train)
    f1 = compute_f1(y_true, y_pred)
    prec = compute_precision(y_true, y_pred)
    rec = compute_recall(y_true, y_pred)
    ni = compute_normalized_improvement(acc, zeror)
 
    return {
        "accuracy": round(acc,   4),
        "zeror": round(zeror, 4),
        "f1_weighted": round(f1,    4),
        "precision_weighted": round(prec,  4),
        "recall_weighted": round(rec,   4),
        "normalized_improvement": round(ni,    4),
    }
 
 
def print_classification_report(y_true, y_pred, label_encoder=None):
    """
    Print a full per-class classification report.
 
    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        label_encoder (LabelEncoder, optional): To display class names.
    """
    target_names = list(label_encoder.classes_) if label_encoder else None
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
 
 
def print_confusion_matrix(y_true, y_pred, label_encoder=None):
    """
    Print confusion matrix.
 
    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        label_encoder (LabelEncoder, optional): To display class names.
    """
    cm = confusion_matrix(y_true, y_pred)
    if label_encoder:
        print("Classes:", list(label_encoder.classes_))
    print(cm)
 