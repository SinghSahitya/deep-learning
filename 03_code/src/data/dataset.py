"""
DeepfakeDataset: PyTorch Dataset for loading preprocessed face images.

Owner: Sahitya

Loads images from CSV split files (train.csv, val.csv, test.csv).
Each CSV has columns: path, label

Returns per item:
    {
        "image": torch.Tensor (3, 224, 224) in [0, 1],
        "label": torch.Tensor scalar (0=real, 1=fake),
        "path": str
    }
"""

import csv
import os

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.augmentations import get_train_transforms, get_val_transforms


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


def get_dataloaders(config):
    """Return train, val, test DataLoaders from config dict."""
    data_config = config["data"]
    image_size = data_config.get("image_size", 224)
    batch_size = config["training"].get("batch_size", 32)
    num_workers = data_config.get("num_workers", 4)

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

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
