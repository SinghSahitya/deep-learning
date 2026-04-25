"""
Preprocess Celeb-DF v2: extract frames, crop faces, balance, split.

Owner: Sahitya

Expected raw directory structure:
    celeb_df_raw/
    ├── Celeb-real/          (real celebrity videos, .mp4)
    ├── Celeb-synthesis/     (deepfake videos, .mp4)
    └── YouTube-real/        (optional, real YouTube videos)

Usage:
    python scripts/preprocess_celebdf.py \
        --input_dir ../04_data/celeb_df_raw \
        --output_dir ../04_data/celeb_df_frames \
        --frame_interval 5
"""

import argparse
import glob
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

from src.data.preprocessing import (
    extract_frames_in_memory,
    crop_faces_batch,
    balance_dataset,
    create_video_splits,
)


def find_videos(input_dir):
    """Find all real and fake video files, return (real_paths, fake_paths)."""
    real_dirs = ["Celeb-real", "YouTube-real"]
    fake_dirs = ["Celeb-synthesis"]

    real_videos = []
    for d in real_dirs:
        pattern = os.path.join(input_dir, d, "*.mp4")
        real_videos.extend(glob.glob(pattern))

    fake_videos = []
    for d in fake_dirs:
        pattern = os.path.join(input_dir, d, "*.mp4")
        fake_videos.extend(glob.glob(pattern))

    return real_videos, fake_videos


def process_videos(video_paths, label, output_dir, detector, frame_interval):
    """Extract frames and crop faces from a list of videos.

    Returns:
        (image_paths, labels, video_ids) — one entry per saved face crop.
    """
    all_paths = []
    all_labels = []
    all_video_ids = []

    for video_path in tqdm(video_paths, desc=f"Processing label={label}"):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(output_dir, f"label_{label}", video_name)

        frames = extract_frames_in_memory(video_path, frame_interval=frame_interval)
        if not frames:
            continue

        saved_paths = crop_faces_batch(frames, video_name, video_output_dir, detector)

        for p in saved_paths:
            all_paths.append(p)
            all_labels.append(label)
            all_video_ids.append(video_name)

    return all_paths, all_labels, all_video_ids


def main():
    parser = argparse.ArgumentParser(description="Preprocess Celeb-DF v2 dataset")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to raw Celeb-DF videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for face crops")
    parser.add_argument("--frame_interval", type=int, default=5, help="Extract every Nth frame (default 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    detector = MTCNN(
        image_size=224,
        margin=44,
        device=device,
        post_process=False,
    )

    # Step 1: Find videos
    print(f"\nScanning {args.input_dir} for videos...")
    real_videos, fake_videos = find_videos(args.input_dir)
    print(f"  Found {len(real_videos)} real videos, {len(fake_videos)} fake videos")

    if len(real_videos) == 0 and len(fake_videos) == 0:
        print("ERROR: No videos found. Check that your input_dir has Celeb-real/ and Celeb-synthesis/ subdirectories.")
        sys.exit(1)

    # Step 2: Extract frames + crop faces
    print("\nExtracting frames and cropping faces...")
    real_paths, real_labels, real_vids = process_videos(
        real_videos, label=0, output_dir=args.output_dir,
        detector=detector, frame_interval=args.frame_interval,
    )
    fake_paths, fake_labels, fake_vids = process_videos(
        fake_videos, label=1, output_dir=args.output_dir,
        detector=detector, frame_interval=args.frame_interval,
    )

    print(f"\n  Real face crops: {len(real_paths)}")
    print(f"  Fake face crops: {len(fake_paths)}")

    # Step 3: Balance dataset
    print("\nBalancing dataset...")
    balanced_real, balanced_fake = balance_dataset(real_paths, fake_paths, seed=args.seed)
    print(f"  Balanced: {len(balanced_real)} real, {len(balanced_fake)} fake")

    # Rebuild aligned lists after balancing
    balanced_real_set = set(balanced_real)
    balanced_fake_set = set(balanced_fake)

    all_paths = []
    all_labels = []
    all_vids = []
    for p, l, v in zip(real_paths, real_labels, real_vids):
        if p in balanced_real_set:
            all_paths.append(p)
            all_labels.append(l)
            all_vids.append(v)
    for p, l, v in zip(fake_paths, fake_labels, fake_vids):
        if p in balanced_fake_set:
            all_paths.append(p)
            all_labels.append(l)
            all_vids.append(v)

    # Step 4: Create video-level splits (no data leakage)
    print("\nCreating video-level train/val/test splits...")
    train_n, val_n, test_n = create_video_splits(
        all_paths, all_labels, all_vids,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    print(f"\n  Train: {train_n} images")
    print(f"  Val:   {val_n} images")
    print(f"  Test:  {test_n} images")
    print(f"\nCSVs saved to {args.output_dir}/{{train,val,test}}.csv")
    print("Done!")


if __name__ == "__main__":
    main()
