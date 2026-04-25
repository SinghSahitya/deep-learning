"""
Simple CNN baseline for deepfake detection.

Owner: Sahitya

Architecture:
    Conv2d(3,32)  -> BN -> ReLU -> MaxPool
    Conv2d(32,64) -> BN -> ReLU -> MaxPool
    Conv2d(64,128)-> BN -> ReLU -> MaxPool
    Conv2d(128,256)-> BN -> ReLU -> AdaptiveAvgPool(1)
    Flatten -> Linear(256, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 1) -> Sigmoid

Accepts (B, 3, H, W) single frames OR (B, T, 3, H, W) clips.
For clips: processes each frame independently through the CNN, then
mean-pools the per-frame features before classification.

forward(x) returns:
    {
        "prediction": (B, 1),
        "spatial_features": (B, 256),
        "freq_features": None
    }
"""

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) single frames  OR  (B, T, 3, H, W) clips
        Returns:
            dict with prediction, spatial_features (B, 256), freq_features=None
        """
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            flat = x.reshape(B * T, C, H, W)
            feat = self.features(flat).flatten(1)       # (B*T, 256)
            feat = feat.reshape(B, T, -1)               # (B, T, 256)
            spatial_features = feat.mean(dim=1)          # (B, 256)
        else:
            feat = self.features(x)                      # (B, 256, 1, 1)
            spatial_features = feat.flatten(1)           # (B, 256)

        prediction = self.classifier(spatial_features)   # (B, 1)

        return {
            "prediction": prediction,
            "spatial_features": spatial_features,
            "freq_features": None,
        }
