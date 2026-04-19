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

import torch
from torch.utils.data import Dataset, DataLoader


class DeepfakeDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        # TODO: Load CSV, store paths and labels
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        # TODO: Load image, apply transform, return dict
        raise NotImplementedError


def get_dataloaders(config):
    """Return train, val, test DataLoaders from config dict."""
    # TODO: Create datasets with appropriate transforms, wrap in DataLoader
    raise NotImplementedError
