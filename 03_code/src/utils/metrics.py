"""
Metrics computation utilities.

Owner: Nitin

Uses sklearn for accuracy, AUC, confusion matrix, precision, recall, F1, ROC curve.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
)


def compute_metrics(predictions, labels, threshold=0.5):
    """
    Compute classification metrics from raw predictions and labels.

    Args:
        predictions: list/array of float probabilities in [0, 1]
        labels: list/array of int ground-truth labels (0 or 1)
        threshold: decision threshold for binary classification

    Returns:
        dict: {accuracy, auc, confusion_matrix, precision, recall, f1, fpr, tpr}
    """
    preds_np = np.array(predictions)
    labels_np = np.array(labels)

    binary_preds = (preds_np >= threshold).astype(int)

    acc = accuracy_score(labels_np, binary_preds)

    try:
        auc = roc_auc_score(labels_np, preds_np)
        fpr, tpr, _ = roc_curve(labels_np, preds_np)
    except ValueError:
        auc = 0.0
        fpr, tpr = np.array([0]), np.array([0])

    cm = confusion_matrix(labels_np, binary_preds)
    prec = precision_score(labels_np, binary_preds, zero_division=0)
    rec = recall_score(labels_np, binary_preds, zero_division=0)
    f1 = f1_score(labels_np, binary_preds, zero_division=0)

    return {
        "accuracy": acc,
        "auc": auc,
        "confusion_matrix": cm,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


def format_results_table(results_dict):
    """
    Format evaluation results as a pandas DataFrame.

    Args:
        results_dict: dict of {condition_name: {metric: value, ...}, ...}
    Returns:
        pd.DataFrame with conditions as rows and metrics as columns
    """
    rows = []
    for condition, metrics in results_dict.items():
        row = {"condition": condition}
        for key, val in metrics.items():
            if key not in ("confusion_matrix", "fpr", "tpr", "predictions", "labels"):
                row[key] = val
        rows.append(row)
    return pd.DataFrame(rows)


def save_results_csv(results_df, path):
    """Save results DataFrame to CSV."""
    results_df.to_csv(path, index=False)
    print(f"Results saved to {path}")
