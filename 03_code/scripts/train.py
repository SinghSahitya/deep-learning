"""
Main training entry point.

Owner: Sahitya (baseline mode), Vishesh (adversarial mode)

Usage:
    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/adversarial_training.yaml
    python scripts/train.py --config configs/multi_domain.yaml
"""

import os
import sys
import argparse
import json

# Add project root to path so `src` is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.models import BaselineCNN, EfficientNetDetector, MultiDomainDetector
from src.training.train_baseline import train_baseline
from src.training.train_adversarial import train_adversarial
from src.data.dataset import DeepfakeDataset

import torch
from torch.utils.data import DataLoader


def create_model(config):
    """Create model based on config model.name."""
    model_name = config.model.name

    if model_name == "baseline":
        return BaselineCNN()
    elif model_name == "efficientnet":
        return EfficientNetDetector(
            spatial_dim=config.model.get("spatial_dim", 256),
            pretrained=True,
        )
    elif model_name == "multi_domain":
        return MultiDomainDetector(
            spatial_dim=config.model.get("spatial_dim", 256),
            freq_dim=config.model.get("freq_dim", 128),
            pretrained=True,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def create_dataloaders(config):
    """Create train and val DataLoaders from config."""
    train_dataset = DeepfakeDataset(
        csv_path=config.data.train_csv,
        root_dir=config.data.root,
        image_size=config.data.image_size,
        augment=True,
    )
    val_dataset = DeepfakeDataset(
        csv_path=config.data.val_csv,
        root_dir=config.data.root,
        image_size=config.data.image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train deepfake detector")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Where to save checkpoints")
    args = parser.parse_args()

    # ── Load config ──
    config = load_config(args.config)
    config["checkpoint_dir"] = args.checkpoint_dir

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"Config: {args.config}")
    print(f"Model: {config.model.name}")
    print(f"Device: {device}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"LR: {config.training.lr}")

    # ── Create model ──
    model = create_model(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ── Create dataloaders ──
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # ── Train ──
    is_adversarial = config.get("adversarial", None) is not None and config.loss.get("lambda_afs", 0) > 0

    if is_adversarial or config.model.name in ("efficientnet", "multi_domain"):
        print("\n═══ Adversarial Training ═══")
        print(f"Lambda AFS: {config.loss.lambda_afs}")
        print(f"Lambda Freq: {config.loss.lambda_freq}")
        print(f"PGD epsilon: {config.adversarial.epsilon}")
        print(f"PGD steps: {config.adversarial.pgd_steps}")
        history = train_adversarial(model, train_loader, val_loader, config, device)
    else:
        print("\n═══ Baseline Training ═══")
        history = train_baseline(model, train_loader, val_loader, config, device)

    # ── Save history ──
    os.makedirs("05_results/logs", exist_ok=True)
    history_path = f"05_results/logs/{config.model.name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")


if __name__ == "__main__":
    main()
