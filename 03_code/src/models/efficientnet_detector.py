"""
EfficientNet-B4 based deepfake detector (spatial branch).

Owner: Sahitya

Architecture:
    Backbone: timm EfficientNet-B4 (pretrained, num_classes=0) -> (B, 1792)
    Feature FC: Linear(1792, 256)
    Classifier: 256->128->1 with ReLU, Dropout, Sigmoid

forward(x) returns:
    {
        "prediction": (B, 1),
        "spatial_features": (B, 256),
        "freq_features": None
    }
"""

import torch.nn as nn


class EfficientNetDetector(nn.Module):
    def __init__(self, spatial_dim=256, pretrained=True):
        super().__init__()
        # TODO: Build backbone (timm), feature_fc, classifier
        raise NotImplementedError

    def forward(self, x):
        # TODO
        raise NotImplementedError

    def freeze_backbone(self):
        # TODO
        raise NotImplementedError

    def unfreeze_backbone(self):
        # TODO
        raise NotImplementedError
