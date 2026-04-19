"""
Visualization utilities for figures and plots.

Owner: Nitin

Generates: confusion matrices, ROC curves, accuracy-vs-epsilon,
           t-SNE embeddings, adversarial example grids, training curves.
"""


def plot_confusion_matrix(cm, title, save_path):
    """Plot 2x2 confusion matrix heatmap (Real/Fake). Save as PNG 300 DPI."""
    # TODO
    raise NotImplementedError


def plot_roc_curves(results_dict, save_path):
    """Plot overlaid ROC curves for multiple models. Include AUC in legend."""
    # TODO
    raise NotImplementedError


def plot_accuracy_vs_epsilon(model_results, save_path):
    """Plot accuracy vs epsilon for FGSM at different strengths."""
    # TODO
    raise NotImplementedError


def plot_tsne(features_clean, features_adv, labels, save_path):
    """t-SNE of clean vs adversarial features, colored by class and condition."""
    # TODO
    raise NotImplementedError


def plot_adversarial_examples(clean_images, adv_images, perturbations, preds_clean, preds_adv, save_path):
    """Grid: [clean | perturbation 10x | adversarial] with prediction captions."""
    # TODO
    raise NotImplementedError


def plot_training_curves(history, save_path):
    """Subplot: train/val loss + val accuracy over epochs."""
    # TODO
    raise NotImplementedError
