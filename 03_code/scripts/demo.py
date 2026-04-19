"""
Gradio demo: upload video/image -> real/fake verdict with confidence.

Owner: Sahitya

Usage:
    python scripts/demo.py --checkpoint checkpoints/multi_domain_full.pth --device cuda

Features:
    - Accept video or image input
    - Extract faces from frames
    - Run model prediction
    - Show per-frame confidence and final verdict
    - Optional adversarial robustness test toggle
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Deepfake detection demo")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    # TODO: Load model, build Gradio interface, launch
    raise NotImplementedError


if __name__ == "__main__":
    main()
