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
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
)


def compute_metrics(predictions, labels, threshold=0.5):
    """
    Compute all classification metrics.

    Args:
        predictions: list/array of sigmoid outputs (floats)
        labels: list/array of ground truth (0 or 1)
        threshold: classification threshold

    Returns:
        dict: {accuracy, auc, confusion_matrix, precision, recall, f1, fpr, tpr}
    """
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Binary predictions from sigmoid outputs
    binary_preds = (predictions >= threshold).astype(int)

    # Core metrics
    accuracy = accuracy_score(labels, binary_preds)

    # AUC — handle edge case where only one class is present
    try:
        auc = roc_auc_score(labels, predictions)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(labels, binary_preds, labels=[0, 1])

    precision = precision_score(labels, binary_preds, zero_division=0)
    recall = recall_score(labels, binary_preds, zero_division=0)
    f1 = f1_score(labels, binary_preds, zero_division=0)

    # ROC curve data
    try:
        fpr, tpr, _ = roc_curve(labels, predictions)
    except ValueError:
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0])

    return {
        "accuracy": accuracy,
        "auc": auc,
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tpr": tpr,
    }


def format_results_table(results_dict):
    """
    Format evaluation results as a pandas DataFrame.

    Args:
        results_dict: dict of {condition_name: metrics_dict, ...}
            Each metrics_dict has keys: accuracy, auc, precision, recall, f1

    Returns:
        pandas DataFrame with rows=conditions, columns=metrics
    """
    rows = []
    for condition, metrics in results_dict.items():
        rows.append(
            {
                "Condition": condition,
                "Accuracy": f"{metrics['accuracy']:.4f}",
                "AUC": f"{metrics['auc']:.4f}",
                "Precision": f"{metrics['precision']:.4f}",
                "Recall": f"{metrics['recall']:.4f}",
                "F1": f"{metrics['f1']:.4f}",
            }
        )

    df = pd.DataFrame(rows)
    df = df.set_index("Condition")
    return df


def save_results_csv(results_df, path):
    """Save results DataFrame to CSV."""
    results_df.to_csv(path, index=True)
