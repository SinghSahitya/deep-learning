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

import torch
import torch.nn as nn
import timm


class EfficientNetDetector(nn.Module):
    def __init__(self, spatial_dim=256, pretrained=True):
        super().__init__()
        self.spatial_dim = spatial_dim

        # Backbone: EfficientNet-B4 with classification head removed
        self.backbone = timm.create_model(
            'efficientnet_b4', pretrained=pretrained, num_classes=0
        )
        # backbone outputs (B, 1792)

        # Project backbone features to spatial_dim
        self.spatial_fc = nn.Linear(1792, spatial_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(spatial_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) in [0, 1]
        Returns:
            dict with prediction, spatial_features, freq_features=None
        """
        backbone_out = self.backbone(x)                  # (B, 1792)
        spatial_features = self.spatial_fc(backbone_out)  # (B, 256)
        prediction = self.classifier(spatial_features)    # (B, 1)

        return {
            "prediction": prediction,
            "spatial_features": spatial_features,
            "freq_features": None,
        }

    def freeze_backbone(self):
        """Freeze EfficientNet backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze EfficientNet backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
