"""
Main training entry point.

Usage:
    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/adversarial_training.yaml
    python scripts/train.py --config configs/multi_domain.yaml

Dispatches to train_baseline() or train_adversarial() based on whether the
config has an "adversarial" section.
"""

import argparse
import json
import os
import sys

import torch
import torch.backends.cudnn as cudnn

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.data.dataset import get_dataloaders
from src.models.baseline_cnn import BaselineCNN
from src.models.efficientnet_detector import EfficientNetDetector
from src.models.multi_domain_model import MultiDomainDetector
from src.training.train_baseline import train_baseline
from src.training.train_adversarial import train_adversarial


def create_model(config):
    name = config.model.name
    if name == "baseline_cnn":
        return BaselineCNN()
    elif name == "efficientnet":
        spatial_dim = config.model.get("spatial_dim", 256)
        return EfficientNetDetector(spatial_dim=spatial_dim, pretrained=True)
    elif name == "multi_domain":
        spatial_dim = config.model.get("spatial_dim", 256)
        freq_dim = config.model.get("freq_dim", 128)
        return MultiDomainDetector(
            spatial_dim=spatial_dim, freq_dim=freq_dim, pretrained=True
        )
    else:
        raise ValueError(f"Unknown model name: {name}")


def _make_json_safe(obj):
    """Recursively convert torch/numpy values in dict/list to python scalars."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(x) for x in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return obj


def main():
    parser = argparse.ArgumentParser(description="Train deepfake detector")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="05_results/models",
        help="Where to save checkpoints",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config["checkpoint_dir"] = args.checkpoint_dir
    config["checkpoint_tag"] = os.path.splitext(os.path.basename(args.config))[0]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    if device == "cuda":
        cudnn.benchmark = True

    print(f"Config: {args.config}")
    print(f"Model: {config.model.name}")
    print(f"Device: {device}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"LR: {config.training.lr}")

    # Create model
    model = create_model(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, _test_loader = get_dataloaders(config)
    print(
        f"Train batches: {len(train_loader)} | "
        f"Val batches: {len(val_loader)} | "
        f"Train samples: {len(train_loader.dataset)} | "
        f"Val samples: {len(val_loader.dataset)}"
    )

    # Dispatch to training mode
    is_adversarial = "adversarial" in config
    if is_adversarial:
        print("\n=== Adversarial Training ===")
        print(f"  lambda_afs:  {config.loss.lambda_afs}")
        print(f"  lambda_freq: {config.loss.lambda_freq}")
        print(f"  epsilon:     {config.adversarial.epsilon}")
        print(f"  PGD steps:   {config.adversarial.pgd_steps}")
        history = train_adversarial(model, train_loader, val_loader, config, device)
    else:
        print("\n=== Baseline Training ===")
        history = train_baseline(model, train_loader, val_loader, config, device)

    # Save history as JSON (use config filename to avoid overwrites)
    logs_dir = "05_results/logs"
    os.makedirs(logs_dir, exist_ok=True)
    config_stem = os.path.splitext(os.path.basename(args.config))[0]
    history_path = os.path.join(logs_dir, f"history_{config_stem}.json")
    with open(history_path, "w") as f:
        json.dump(_make_json_safe(history), f, indent=2)
    print(f"\nTraining history saved to {history_path}")


if __name__ == "__main__":
    main()
