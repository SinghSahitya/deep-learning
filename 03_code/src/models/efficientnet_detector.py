"""
EfficientNet-B4 based deepfake detector (spatial branch).

Owner: Sahitya

Architecture:
    Backbone: timm EfficientNet-B4 (pretrained, num_classes=0) -> (B, 1792)
    Feature FC: Linear(1792, spatial_dim) -> (B, 256)
    Classifier: 256 -> 128 -> 1 with ReLU, Dropout, Sigmoid

Accepts (B, 3, H, W) single frames OR (B, T, 3, H, W) clips.
For clips: processes each frame through EfficientNet independently, then
mean-pools the per-frame features before classification.

forward(x) returns:
    {
        "prediction": (B, 1) sigmoid,
        "spatial_features": (B, spatial_dim),
        "freq_features": None
    }

NOTE: The projection layer is named `feature_fc` to maintain compatibility
with baseline checkpoints trained with this name.
"""

import torch
import torch.nn as nn
import timm


class EfficientNetDetector(nn.Module):
    def __init__(self, spatial_dim=256, pretrained=True):
        super().__init__()
        self.spatial_dim = spatial_dim

        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=pretrained, num_classes=0
        )

        self.feature_fc = nn.Linear(1792, spatial_dim)

        self.classifier = nn.Sequential(
            nn.Linear(spatial_dim, 128),
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
            dict with prediction, spatial_features, freq_features=None
        """
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            flat = x.reshape(B * T, C, H, W)
            backbone_out = self.backbone(flat)               # (B*T, 1792)
            projected = self.feature_fc(backbone_out)        # (B*T, spatial_dim)
            projected = projected.reshape(B, T, -1)          # (B, T, spatial_dim)
            spatial_features = projected.mean(dim=1)         # (B, spatial_dim)
        else:
            backbone_features = self.backbone(x)             # (B, 1792)
            spatial_features = self.feature_fc(backbone_features)

        prediction = self.classifier(spatial_features)       # (B, 1)

        return {
            "prediction": prediction,
            "spatial_features": spatial_features,
            "freq_features": None,
        }

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
