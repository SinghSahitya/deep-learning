"""
Baseline CNN classifier for deepfake detection.

Owner: Sahitya

Architecture:
    4 conv blocks (3->32->64->128->256) with BN, ReLU, pooling
    FC head: 256->128->1 with dropout

forward(x) returns:
    {
        "prediction": (B, 1) sigmoid output,
        "spatial_features": (B, 256) features before FC head,
        "freq_features": None
    }
"""

import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: (B, 3, 224, 224) -> (B, 32, 112, 112)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: (B, 32, 112, 112) -> (B, 64, 56, 56)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: (B, 64, 56, 56) -> (B, 128, 28, 28)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: (B, 128, 28, 28) -> (B, 256, 1, 1)
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
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) images in [0, 1]
        Returns:
            dict with prediction, spatial_features, freq_features=None
        """
        feat = self.features(x)                    # (B, 256, 1, 1)
        spatial_features = feat.flatten(1)          # (B, 256)
        prediction = self.classifier(spatial_features)  # (B, 1)

        return {
            "prediction": prediction,
            "spatial_features": spatial_features,
            "freq_features": None,
        }
