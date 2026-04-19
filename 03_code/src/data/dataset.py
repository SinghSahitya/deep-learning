import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from .augmentations import get_train_transforms, get_val_transforms

class DeepfakeDataset(Dataset):
    """
    Args:
        csv_path: path to train.csv / val.csv / test.csv
        transform: torchvision transforms (optional)
        
    Each CSV has columns: path, label
    
    __getitem__ returns:
        {
            "image": torch.Tensor (3, 224, 224) in [0, 1],
            "label": torch.Tensor scalar (0 or 1),
            "path": str
        }
    """
    def __init__(self, csv_path, transform=None):
        self.data_info = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data_info.iloc[idx, 0]
        # Label: 0 = real, 1 = fake
        label = self.data_info.iloc[idx, 1]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback for unexpected missing files
            print(f"Error loading image {img_path}: {e}")
            # Instead of crashing, load another random image
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        # Ensure label is float for BCEWithLogitsLoss or BCELoss with Sigmoid
        label = torch.tensor(label, dtype=torch.float32)

        sample = {
            "image": image,
            "label": label.unsqueeze(0), # Shape: (1,) for BCELoss
            "path": img_path
        }
        return sample

def get_dataloaders(config):
    """
    Helper function to build train/val/test dataloaders from config dict.
    """
    batch_size = config.get('training', {}).get('batch_size', 32)
    num_workers = config.get('data', {}).get('num_workers', 4)
    image_size = config.get('data', {}).get('image_size', 224)
    
    train_csv = config.get('data', {}).get('train_csv', '')
    val_csv = config.get('data', {}).get('val_csv', '')
    test_csv = config.get('data', {}).get('test_csv', '')
    
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)
    
    train_dataset = DeepfakeDataset(train_csv, transform=train_transform)
    val_dataset = DeepfakeDataset(val_csv, transform=val_transform)
    test_dataset = DeepfakeDataset(test_csv, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
