"""
Visualization utilities for figures and plots.

Owner: Nitin

Generates: confusion matrices, ROC curves, accuracy-vs-epsilon,
           t-SNE embeddings, adversarial example grids, training curves.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, title, save_path):
    """Plot 2x2 confusion matrix heatmap (Real/Fake). Save as PNG 300 DPI."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'], ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close(fig)


def plot_roc_curves(results_dict, save_path):
    """Plot overlaid ROC curves for multiple models. Include AUC in legend."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, m in results_dict.items():
        ax.plot(m.get("fpr", [0, 1]), m.get("tpr", [0, 1]), label=f"{name} (AUC={m.get('auc', 0):.4f})")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC Curves')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close(fig)


def plot_accuracy_vs_epsilon(model_results, save_path):
    """Plot accuracy vs epsilon for FGSM at different strengths."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, eps_acc in model_results.items():
        epsilons = sorted(eps_acc.keys())
        ax.plot(epsilons, [eps_acc[e] for e in epsilons], 'o-', label=model_name)
    ax.set_xlabel('Epsilon'); ax.set_ylabel('Accuracy'); ax.set_title('Accuracy vs Epsilon')
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close(fig)


def plot_tsne(features_clean, features_adv, labels, save_path):
    """t-SNE of clean vs adversarial features, colored by class and condition."""
    from sklearn.manifold import TSNE
    all_features = np.vstack([features_clean, features_adv])
    n = len(features_clean)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(all_features)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(embedded[:n, 0], embedded[:n, 1], c='blue', alpha=0.5, s=10, label='Clean')
    axes[0].scatter(embedded[n:, 0], embedded[n:, 1], c='red', alpha=0.5, s=10, label='Adversarial')
    axes[0].legend(); axes[0].set_title('By Condition')
    labels_all = list(labels) + list(labels)
    for lv, c, nm in [(0, 'green', 'Real'), (1, 'orange', 'Fake')]:
        mask = [l == lv for l in labels_all]
        axes[1].scatter(embedded[mask, 0], embedded[mask, 1], c=c, alpha=0.5, s=10, label=nm)
    axes[1].legend(); axes[1].set_title('By Class')
    plt.suptitle('t-SNE Features'); plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close(fig)


def plot_adversarial_examples(clean_images, adv_images, perturbations, preds_clean, preds_adv, save_path):
    """Grid: [clean | perturbation 10x | adversarial] with prediction captions."""
    n = min(len(clean_images), 5)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1: axes = axes[np.newaxis, :]
    for i in range(n):
        axes[i, 0].imshow(np.clip(clean_images[i].transpose(1, 2, 0), 0, 1))
        axes[i, 0].set_title(f"Clean ({preds_clean[i]:.3f})"); axes[i, 0].axis('off')
        pert = perturbations[i].transpose(1, 2, 0) * 10
        pert = (pert - pert.min()) / (pert.max() - pert.min() + 1e-8)
        axes[i, 1].imshow(pert); axes[i, 1].set_title("Perturbation (10x)"); axes[i, 1].axis('off')
        axes[i, 2].imshow(np.clip(adv_images[i].transpose(1, 2, 0), 0, 1))
        axes[i, 2].set_title(f"Adversarial ({preds_adv[i]:.3f})"); axes[i, 2].axis('off')
    plt.suptitle('Adversarial Examples'); plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close(fig)


def plot_training_curves(history, save_path):
    """Subplot: train/val loss + val accuracy over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history["train_loss"], 'b-', label='Train Loss')
    if "val_loss" in history: axes[0].plot(epochs, history["val_loss"], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    if "val_clean_acc" in history: axes[1].plot(epochs, history["val_clean_acc"], 'g-', label='Clean Acc')
    if "val_pgd_acc" in history: axes[1].plot(epochs, history["val_pgd_acc"], 'm-', label='PGD Acc')
    elif "val_acc" in history: axes[1].plot(epochs, history["val_acc"], 'g-', label='Val Acc')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy'); axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.suptitle('Training Curves'); plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close(fig)
