"""
Multi-domain deepfake detector: spatial (EfficientNet-B4) + frequency (FFT CNN).

Owner: Vishesh

Architecture:
    Spatial: EfficientNet-B4 -> (B, 256)
    Frequency: FrequencyBranch -> (B, 128)
    Fusion: cat(256, 128) = 384 -> FC(384,128) -> ReLU -> Dropout -> FC(128,1) -> Sigmoid

forward(x) returns:
    {
        "prediction": (B, 1),
        "spatial_features": (B, 256),
        "freq_features": (B, 128)
    }
"""

import torch
import torch.nn as nn
import timm
from .frequency_branch import FrequencyBranch


class MultiDomainDetector(nn.Module):
    """
    Dual-branch deepfake detector combining spatial and frequency features.

    The spatial branch uses a pretrained EfficientNet-B4 backbone, while the
    frequency branch applies FFT-based feature extraction. Both branches
    produce features independently, which are then concatenated and fed
    through a fusion classifier.
    """

    def __init__(self, spatial_dim=256, freq_dim=128, pretrained=True):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.freq_dim = freq_dim

        # ── Spatial branch: EfficientNet-B4 ──
        self.backbone = timm.create_model(
            'efficientnet_b4', pretrained=pretrained, num_classes=0
        )
        # backbone outputs (B, 1792)
        self.spatial_fc = nn.Linear(1792, spatial_dim)  # (B, 256)

        # ── Frequency branch ──
        self.freq_branch = FrequencyBranch(output_dim=freq_dim)  # (B, 128)

        # ── Fusion classifier ──
        self.classifier = nn.Sequential(
            nn.Linear(spatial_dim + freq_dim, 128),  # 384 -> 128
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
            dict with prediction (B,1), spatial_features (B,256), freq_features (B,128)
        """
        # Spatial branch
        backbone_out = self.backbone(x)                  # (B, 1792)
        spatial_features = self.spatial_fc(backbone_out)  # (B, spatial_dim)

        # Frequency branch
        freq_features = self.freq_branch(x)               # (B, freq_dim)

        # Fusion
        combined = torch.cat([spatial_features, freq_features], dim=1)  # (B, 384)
        prediction = self.classifier(combined)             # (B, 1)

        return {
            "prediction": prediction,
            "spatial_features": spatial_features,
            "freq_features": freq_features,
        }

    def freeze_backbone(self):
        """Freeze EfficientNet backbone parameters for warm-up training."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze EfficientNet backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
