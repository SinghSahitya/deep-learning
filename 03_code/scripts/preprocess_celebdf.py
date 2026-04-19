"""
Preprocess Celeb-DF v2: extract frames, crop faces, balance, split.

Owner: Sahitya

Usage:
    python scripts/preprocess_celebdf.py \
        --input_dir data/celeb_df_raw \
        --output_dir data/celeb_df_frames \
        --frame_interval 10
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Preprocess Celeb-DF v2 dataset")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to raw Celeb-DF videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for face crops")
    parser.add_argument("--frame_interval", type=int, default=10, help="Extract every Nth frame")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # TODO: Extract frames, crop faces, balance, create splits
    raise NotImplementedError


if __name__ == "__main__":
    main()
