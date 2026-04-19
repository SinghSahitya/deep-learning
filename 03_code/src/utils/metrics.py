"""
Metrics computation utilities.

Owner: Nitin

Uses sklearn for accuracy, AUC, confusion matrix, precision, recall, F1, ROC curve.
"""


def compute_metrics(predictions, labels, threshold=0.5):
    """
    Returns:
        dict: {accuracy, auc, confusion_matrix, precision, recall, f1, fpr, tpr}
    """
    # TODO
    raise NotImplementedError


def format_results_table(results_dict):
    """Format evaluation results as a pandas DataFrame."""
    # TODO
    raise NotImplementedError


def save_results_csv(results_df, path):
    """Save results DataFrame to CSV."""
    # TODO
    raise NotImplementedError
