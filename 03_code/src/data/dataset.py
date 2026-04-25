"""
DeepfakeDataset: PyTorch Dataset for loading preprocessed face images.
DeepfakeVideoDataset: PyTorch Dataset that loads clips of T consecutive
                      frames from the same video, preserving temporal info.

Owner: Sahitya

CSV format: path, label[, video_id]

DeepfakeDataset returns per item:
    {"image": (3,H,W), "label": scalar, "path": str}

DeepfakeVideoDataset returns per item:
    {"clip": (T,3,H,W), "label": scalar, "video_id": str}
"""

import csv
import os
import re
from collections import defaultdict

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.augmentations import (
    get_train_transforms,
    get_val_transforms,
    ClipTrainTransform,
    ClipValTransform,
)


class DeepfakeDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.csv_path = csv_path
        self.transform = transform or get_val_transforms()
        self.samples = []

        csv_dir = os.path.dirname(os.path.abspath(csv_path))
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = row["path"]
                if not os.path.isabs(image_path):
                    image_path = os.path.abspath(os.path.join(csv_dir, image_path))

                self.samples.append(
                    {
                        "path": image_path,
                        "label": int(row["label"]),
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample["path"]).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(sample["label"], dtype=torch.long)

        return {
            "image": image,
            "label": label,
            "path": sample["path"],
        }


def _video_id_from_path(path):
    """Extract video id from a frame path like '.../videoname_frame_30.png'."""
    basename = os.path.splitext(os.path.basename(path))[0]
    match = re.match(r"^(.+)_frame_\d+$", basename)
    if match:
        return match.group(1)
    return basename


class DeepfakeVideoDataset(Dataset):
    """
    Groups frames by video and returns fixed-length clips of T consecutive
    frames.  Each sample is one clip from one video.

    If a video has fewer than T frames, the last frame is repeated.
    If a video has more than T frames, multiple non-overlapping clips are
    created (with the last clip padded if necessary).
    """

    def __init__(self, csv_path, clip_length=16, transform=None):
        self.clip_length = clip_length
        self.transform = transform or ClipValTransform()

        csv_dir = os.path.dirname(os.path.abspath(csv_path))
        video_frames = defaultdict(list)
        video_labels = {}

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = row["path"]
                if not os.path.isabs(image_path):
                    image_path = os.path.abspath(os.path.join(csv_dir, image_path))

                vid = row.get("video_id") or _video_id_from_path(image_path)
                video_frames[vid].append(image_path)
                video_labels[vid] = int(row["label"])

        self.clips = []
        for vid, paths in video_frames.items():
            paths.sort()
            label = video_labels[vid]
            T = clip_length

            if len(paths) <= T:
                self.clips.append((paths, label, vid))
            else:
                for start in range(0, len(paths), T):
                    chunk = paths[start : start + T]
                    self.clips.append((chunk, label, vid))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        paths, label, vid = self.clips[idx]
        T = self.clip_length

        frames = [Image.open(p).convert("RGB") for p in paths]
        while len(frames) < T:
            frames.append(frames[-1].copy())

        clip = self.transform(frames)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "clip": clip,
            "label": label_tensor,
            "video_id": vid,
        }


def get_dataloaders(config):
    """Return train, val, test DataLoaders from config dict.

    When config.data.clip_length > 0 (or is present), uses
    DeepfakeVideoDataset that returns clips of T frames.
    Otherwise falls back to the original single-frame DeepfakeDataset.
    """
    data_config = config["data"]
    image_size = data_config.get("image_size", 224)
    batch_size = config["training"].get("batch_size", 32)
    num_workers = data_config.get("num_workers", 4)
    clip_length = data_config.get("clip_length", 0)

    pin_memory = torch.cuda.is_available()
    persistent = num_workers > 0
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    if clip_length > 0:
        train_transform = ClipTrainTransform(image_size)
        val_transform = ClipValTransform(image_size)

        train_dataset = DeepfakeVideoDataset(
            csv_path=data_config["train_csv"],
            clip_length=clip_length,
            transform=train_transform,
        )
        val_dataset = DeepfakeVideoDataset(
            csv_path=data_config["val_csv"],
            clip_length=clip_length,
            transform=val_transform,
        )
        test_dataset = DeepfakeVideoDataset(
            csv_path=data_config["test_csv"],
            clip_length=clip_length,
            transform=val_transform,
        )
    else:
        train_dataset = DeepfakeDataset(
            csv_path=data_config["train_csv"],
            transform=get_train_transforms(image_size),
        )
        val_dataset = DeepfakeDataset(
            csv_path=data_config["val_csv"],
            transform=get_val_transforms(image_size),
        )
        test_dataset = DeepfakeDataset(
            csv_path=data_config["test_csv"],
            transform=get_val_transforms(image_size),
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs,
    )

    return train_loader, val_loader, test_loader
