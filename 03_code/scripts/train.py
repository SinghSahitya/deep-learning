"""
Main training entry point.

Owner: Sahitya (baseline mode), Vishesh (adversarial mode)

Usage:
    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/adversarial_training.yaml
    python scripts/train.py --config configs/multi_domain.yaml
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Train deepfake detector")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Where to save checkpoints")
    args = parser.parse_args()

    # TODO: Load config, create dataloaders, create model, train
    raise NotImplementedError


if __name__ == "__main__":
    main()
