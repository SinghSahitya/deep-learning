"""
Visualization utilities for figures and plots.

Owner: Nitin

Generates: confusion matrices, ROC curves, accuracy-vs-epsilon,
           t-SNE embeddings, adversarial example grids, training curves.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE


# Clean, publication-ready style
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    }
)


def _ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def plot_confusion_matrix(cm, title, save_path):
    """
    Plot a 2x2 confusion matrix heatmap.
    Labels: ["Real", "Fake"] on both axes.
    """
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        ax=ax,
        cbar=True,
        square=True,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(results_dict, save_path):
    """
    Plot ROC curves for multiple models/conditions on the same figure.

    Args:
        results_dict: {model_name: {"fpr": array, "tpr": array, "auc": float}, ...}
    """
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.tab10.colors
    for i, (name, data) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        auc_val = data["auc"]
        ax.plot(
            data["fpr"],
            data["tpr"],
            color=color,
            lw=2,
            label=f"{name} (AUC = {auc_val:.3f})",
        )

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_vs_epsilon(model_results, save_path):
    """
    Plot accuracy vs epsilon for FGSM attacks across multiple models.

    Args:
        model_results: dict of {model_name: {epsilon_value: accuracy, ...}, ...}
    """
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.tab10.colors
    for i, (model_name, eps_acc) in enumerate(model_results.items()):
        epsilons = sorted(eps_acc.keys())
        accuracies = [eps_acc[e] for e in epsilons]
        color = colors[i % len(colors)]
        ax.plot(
            range(len(epsilons)),
            accuracies,
            marker="o",
            color=color,
            lw=2,
            markersize=8,
            label=model_name,
        )

    epsilons_ref = sorted(list(model_results.values())[0].keys())
    eps_labels = []
    for e in epsilons_ref:
        if e == 0:
            eps_labels.append("0")
        else:
            val = round(e * 255)
            eps_labels.append(f"{val}/255")

    ax.set_xticks(range(len(epsilons_ref)))
    ax.set_xticklabels(eps_labels)
    ax.set_xlabel("Epsilon (FGSM)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Perturbation Budget")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tsne(features_clean, features_adv, labels, save_path):
    """
    t-SNE visualization of feature embeddings (clean vs adversarial, by class).
    """
    _ensure_dir(save_path)
    features_clean = np.array(features_clean)
    features_adv = np.array(features_adv)
    labels = np.array(labels)

    N = len(labels)
    combined = np.vstack([features_clean, features_adv])

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(N - 1, 2)))
    embedded = tsne.fit_transform(combined)

    clean_emb = embedded[:N]
    adv_emb = embedded[N:]

    fig, ax = plt.subplots(figsize=(10, 8))

    categories = [
        ("Real (Clean)", labels == 0, clean_emb, "tab:blue", "o"),
        ("Fake (Clean)", labels == 1, clean_emb, "tab:orange", "o"),
        ("Real (Adversarial)", labels == 0, adv_emb, "tab:cyan", "x"),
        ("Fake (Adversarial)", labels == 1, adv_emb, "tab:red", "x"),
    ]

    for name, mask, emb, color, marker in categories:
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            c=color,
            marker=marker,
            label=name,
            alpha=0.6,
            s=20,
        )

    ax.set_title("t-SNE Feature Embeddings")
    ax.legend(fontsize=10, markerscale=2)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_adversarial_examples(
    clean_images, adv_images, perturbations, preds_clean, preds_adv, save_path
):
    """
    Grid of [Clean | Perturbation 10x | Adversarial] rows with prediction captions.

    Accepts (N, C, H, W) or (N, H, W, C) arrays.
    """
    _ensure_dir(save_path)
    clean_images = np.array(clean_images)
    adv_images = np.array(adv_images)
    perturbations = np.array(perturbations)

    # Convert (N, C, H, W) -> (N, H, W, C) if needed
    if clean_images.ndim == 4 and clean_images.shape[1] == 3:
        clean_images = clean_images.transpose(0, 2, 3, 1)
        adv_images = adv_images.transpose(0, 2, 3, 1)
        perturbations = perturbations.transpose(0, 2, 3, 1)

    num_examples = min(6, len(clean_images))
    fig, axes = plt.subplots(num_examples, 3, figsize=(12, 3 * num_examples))
    if num_examples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Clean Image", "Perturbation (10x)", "Adversarial Image"]

    for i in range(num_examples):
        axes[i, 0].imshow(np.clip(clean_images[i], 0, 1))
        axes[i, 0].axis("off")

        pert_display = perturbations[i] * 10
        pert_display = (pert_display - pert_display.min()) / (
            pert_display.max() - pert_display.min() + 1e-8
        )
        axes[i, 1].imshow(pert_display)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(np.clip(adv_images[i], 0, 1))
        axes[i, 2].axis("off")

        clean_label = "Real" if preds_clean[i] < 0.5 else "Fake"
        adv_label = "Real" if preds_adv[i] < 0.5 else "Fake"
        axes[i, 0].set_title(
            f"Clean: {clean_label}({preds_clean[i]:.2f})", fontsize=10
        )
        axes[i, 2].set_title(
            f"Adv: {adv_label}({preds_adv[i]:.2f})", fontsize=10
        )

    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=12, fontweight="bold", pad=10)

    plt.suptitle("Adversarial Example Visualization", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history, save_path):
    """
    Plot training loss curves and accuracy over epochs.

    Supports both baseline history (val_acc) and adversarial history
    (val_clean_acc, val_pgd_acc).
    """
    _ensure_dir(save_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", lw=2, label="Train Loss")
    if "val_loss" in history:
        ax1.plot(epochs, history["val_loss"], "r-", lw=2, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    plotted_acc = False
    if "val_clean_acc" in history:
        ax2.plot(epochs, history["val_clean_acc"], "g-", lw=2, label="Val Clean Acc")
        plotted_acc = True
    if "val_pgd_acc" in history:
        ax2.plot(epochs, history["val_pgd_acc"], "m-", lw=2, label="Val PGD Acc")
        plotted_acc = True
    if not plotted_acc and "val_acc" in history:
        ax2.plot(epochs, history["val_acc"], "g-", lw=2, label="Val Accuracy")
        plotted_acc = True

    if plotted_acc:
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Validation Accuracy")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0.0, 1.05])
    else:
        ax2.text(0.5, 0.5, "No validation accuracy data", ha="center", va="center")
        ax2.set_title("Validation Accuracy")

    plt.suptitle("Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
