"""
Evaluation entry point: run clean + adversarial evaluations and save results.

Owner: Nitin

Usage:
    python scripts/eval.py \
        --checkpoint checkpoints/multi_domain_full.pth \
        --config configs/multi_domain.yaml \
        --model_name "Full Model (Ours)" \
        --output_dir ../05_results/ \
        --run_autoattack \
        --autoattack_samples 500
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detector")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--model_name", type=str, required=True, help="Name for results table")
    parser.add_argument("--output_dir", type=str, default="../05_results/", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_autoattack", action="store_true", help="Run AutoAttack (slow)")
    parser.add_argument("--autoattack_samples", type=int, default=500)
    args = parser.parse_args()

    # TODO: Load model, load test data, run evaluations, save results and figures
    raise NotImplementedError


if __name__ == "__main__":
    main()
