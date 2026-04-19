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
        # TODO: Build conv blocks and classifier
        raise NotImplementedError

    def forward(self, x):
        # TODO: Return dict with prediction, spatial_features, freq_features=None
        raise NotImplementedError
