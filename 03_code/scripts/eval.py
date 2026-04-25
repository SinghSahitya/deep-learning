"""
Evaluation entry point: run clean + adversarial evaluations and save results.

Usage:
    python scripts/eval.py \
        --checkpoint 05_results/models/best_multi_domain.pth \
        --config configs/multi_domain.yaml \
        --model_name "Full Model (Ours)" \
        --output_dir ../05_results/ \
        --run_autoattack \
        --autoattack_samples 500
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.training.evaluate import evaluate_model
from src.utils.metrics import compute_metrics, format_results_table, save_results_csv
from src.utils.visualization import (
    plot_adversarial_examples,
    plot_confusion_matrix,
    plot_roc_curves,
)


def load_model(config, checkpoint_path, device):
    """Load model from config and checkpoint."""
    model_name = config["model"]["name"]

    if model_name == "baseline_cnn":
        from src.models.baseline_cnn import BaselineCNN

        model = BaselineCNN()
    elif model_name == "efficientnet":
        from src.models.efficientnet_detector import EfficientNetDetector

        spatial_dim = config["model"].get("spatial_dim", 256)
        model = EfficientNetDetector(spatial_dim=spatial_dim, pretrained=False)
    elif model_name == "multi_domain":
        from src.models.multi_domain_model import MultiDomainDetector

        spatial_dim = config["model"].get("spatial_dim", 256)
        freq_dim = config["model"].get("freq_dim", 128)
        model = MultiDomainDetector(
            spatial_dim=spatial_dim, freq_dim=freq_dim, pretrained=False
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def load_test_data(config):
    """Load test dataset and create DataLoader."""
    from src.data.dataset import DeepfakeDataset
    from src.data.augmentations import get_val_transforms

    test_csv = config["data"]["test_csv"]
    image_size = config["data"].get("image_size", 224)
    num_workers = config["data"].get("num_workers", 4)
    batch_size = config["training"]["batch_size"]

    test_dataset = DeepfakeDataset(
        csv_path=test_csv,
        transform=get_val_transforms(image_size),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return test_dataset, test_loader


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detector")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../05_results/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_autoattack", action="store_true")
    parser.add_argument("--autoattack_samples", type=int, default=500)
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    print(f"Using device: {device}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    figures_dir = os.path.join(args.output_dir, "figures")
    logs_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print(f"Loading model from: {args.checkpoint}")
    model = load_model(config, args.checkpoint, device)

    print("Loading test data...")
    test_dataset, test_loader = load_test_data(config)

    attacks = {
        "FGSM (eps=2/255)": lambda m, x, y: fgsm_attack(m, x, y, epsilon=2 / 255),
        "FGSM (eps=4/255)": lambda m, x, y: fgsm_attack(m, x, y, epsilon=4 / 255),
        "FGSM (eps=8/255)": lambda m, x, y: fgsm_attack(m, x, y, epsilon=8 / 255),
        "PGD (eps=4/255)": lambda m, x, y: pgd_attack(
            m, x, y, epsilon=4 / 255, num_steps=10, alpha=1 / 255
        ),
    }

    print(f"\n=== Evaluating: {args.model_name} ===")
    results = evaluate_model(model, test_loader, device, attacks=attacks)

    print(f"\n--- Results for {args.model_name} ---")
    for condition, metrics in results.items():
        print(
            f"  {condition:25s}  Acc: {metrics['accuracy']:.4f}  "
            f"AUC: {metrics['auc']:.4f}  F1: {metrics['f1']:.4f}"
        )

    # Save / append results row to main_results.csv
    results_csv_path = os.path.join(args.output_dir, "main_results.csv")
    row = {"Model": args.model_name}
    for condition, metrics in results.items():
        row[f"{condition} Acc"] = round(float(metrics["accuracy"]), 4)
        row[f"{condition} AUC"] = round(float(metrics["auc"]), 4)

    import pandas as pd

    if os.path.exists(results_csv_path) and os.path.getsize(results_csv_path) > 0:
        try:
            existing = pd.read_csv(results_csv_path)
            # Remove existing row with same model name (re-runs overwrite)
            existing = existing[existing["Model"] != args.model_name]
            df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        except Exception:
            df = pd.DataFrame([row])
    else:
        df = pd.DataFrame([row])
    df.to_csv(results_csv_path, index=False)
    print(f"\nResults saved to: {results_csv_path}")

    # Confusion matrices
    safe_name = (
        args.model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    )
    for condition, metrics in results.items():
        safe_cond = (
            condition.lower().replace(" ", "_").replace("/", "_").replace("=", "")
        )
        cm_path = os.path.join(
            figures_dir, f"confusion_matrix_{safe_name}_{safe_cond}.png"
        )
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            title=f"{args.model_name} — {condition}",
            save_path=cm_path,
        )
    print(f"Confusion matrices saved to: {figures_dir}")

    # ROC curves
    roc_data = {}
    for condition, metrics in results.items():
        roc_data[f"{args.model_name} ({condition})"] = {
            "fpr": metrics["fpr"],
            "tpr": metrics["tpr"],
            "auc": metrics["auc"],
        }
    roc_path = os.path.join(figures_dir, f"roc_curves_{safe_name}.png")
    plot_roc_curves(roc_data, roc_path)
    print(f"ROC curves saved to: {roc_path}")

    # Adversarial example visualization (using PGD)
    print("\nGenerating adversarial example visualizations...")
    model.eval()
    sample_batch = next(iter(test_loader))

    if "clip" in sample_batch:
        sample_inputs = sample_batch["clip"][:6].to(device)
    else:
        sample_inputs = sample_batch["image"][:6].to(device)

    sample_labels_raw = sample_batch["label"][:6].to(device)
    if sample_labels_raw.dim() > 1:
        sample_labels = sample_labels_raw.squeeze(-1).long()
    else:
        sample_labels = sample_labels_raw.long()

    adv_inputs = pgd_attack(model, sample_inputs, sample_labels, epsilon=4 / 255)
    perturbations = adv_inputs - sample_inputs

    with torch.no_grad():
        clean_preds = model(sample_inputs)["prediction"].squeeze(1).cpu().numpy()
        adv_preds = model(adv_inputs)["prediction"].squeeze(1).cpu().numpy()

    # For visualization: use first frame of each clip if 5D, otherwise use as-is
    vis_clean = sample_inputs
    vis_adv = adv_inputs
    vis_pert = perturbations
    if vis_clean.dim() == 5:
        vis_clean = vis_clean[:, 0]
        vis_adv = vis_adv[:, 0]
        vis_pert = vis_pert[:, 0]

    adv_vis_path = os.path.join(figures_dir, f"adversarial_examples_{safe_name}.png")
    plot_adversarial_examples(
        vis_clean.cpu().numpy(),
        vis_adv.cpu().numpy(),
        vis_pert.cpu().numpy(),
        clean_preds,
        adv_preds,
        adv_vis_path,
    )
    print(f"Adversarial examples saved to: {adv_vis_path}")

    # Optional: AutoAttack evaluation
    if args.run_autoattack:
        print(f"\nRunning AutoAttack on {args.autoattack_samples} samples...")
        from src.attacks.auto_attack import auto_attack_eval

        subset_size = min(args.autoattack_samples, len(test_dataset))
        subset_indices = list(range(subset_size))
        subset = Subset(test_dataset, subset_indices)
        subset_loader = DataLoader(subset, batch_size=subset_size, shuffle=False)
        subset_batch = next(iter(subset_loader))

        aa_images = subset_batch["image"].to(device)
        aa_labels_raw = subset_batch["label"].to(device)
        if aa_labels_raw.dim() > 1:
            aa_labels = aa_labels_raw.squeeze(-1).long()
        else:
            aa_labels = aa_labels_raw.long()

        aa_results = auto_attack_eval(
            model, aa_images, aa_labels, epsilon=4 / 255, batch_size=32
        )
        print(
            f"  AutoAttack — Clean Acc: {aa_results['clean_accuracy']:.4f}, "
            f"Robust Acc: {aa_results['robust_accuracy']:.4f}"
        )
        results["AutoAttack"] = {
            "accuracy": aa_results["robust_accuracy"],
            "clean_accuracy": aa_results["clean_accuracy"],
        }

    # Save full evaluation log as JSON
    log_path = os.path.join(logs_dir, f"eval_{safe_name}.json")
    log_data = {}
    for condition, metrics in results.items():
        log_data[condition] = {
            "accuracy": float(metrics.get("accuracy", 0)),
            "auc": float(metrics.get("auc", 0)),
            "precision": float(metrics.get("precision", 0))
            if "precision" in metrics
            else None,
            "recall": float(metrics.get("recall", 0)) if "recall" in metrics else None,
            "f1": float(metrics.get("f1", 0)) if "f1" in metrics else None,
        }

    with open(log_path, "w") as f:
        json.dump({"model_name": args.model_name, "results": log_data}, f, indent=2)
    print(f"Evaluation log saved to: {log_path}")

    print(f"\n=== Evaluation complete for {args.model_name} ===")


if __name__ == "__main__":
    main()
