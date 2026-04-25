"""
Multi-domain deepfake detector: spatial (EfficientNet-B4) + frequency (3D FFT CNN).

Owner: Vishesh

Accepts (B, 3, H, W) single frames OR (B, T, 3, H, W) video clips.

Architecture:
    Spatial: EfficientNet-B4 per frame -> mean pool over T -> (B, 256)
    Frequency: FrequencyBranch with 3D FFT over (T, H, W) -> (B, 128)
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

    Spatial branch: EfficientNet-B4 processes each frame independently, then
    mean-pools over the temporal dimension.

    Frequency branch: 3D FFT over (T, H, W) captures temporal flickering +
    spatial artifacts simultaneously.  Falls back to 2D for single frames.
    """

    def __init__(self, spatial_dim=256, freq_dim=128, pretrained=True):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.freq_dim = freq_dim

        self.backbone = timm.create_model(
            'efficientnet_b4', pretrained=pretrained, num_classes=0
        )
        self.spatial_fc = nn.Linear(1792, spatial_dim)

        self.freq_branch = FrequencyBranch(output_dim=freq_dim)

        self.classifier = nn.Sequential(
            nn.Linear(spatial_dim + freq_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, 3, H, W) clip  OR  (B, 3, H, W) single frame
        Returns:
            dict with prediction (B,1), spatial_features (B,256), freq_features (B,128)
        """
        # ── Spatial branch ──
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            flat = x.reshape(B * T, C, H, W)
            backbone_out = self.backbone(flat)               # (B*T, 1792)
            projected = self.spatial_fc(backbone_out)        # (B*T, spatial_dim)
            projected = projected.reshape(B, T, -1)          # (B, T, spatial_dim)
            spatial_features = projected.mean(dim=1)         # (B, spatial_dim)
        else:
            backbone_out = self.backbone(x)                  # (B, 1792)
            spatial_features = self.spatial_fc(backbone_out)  # (B, spatial_dim)

        # ── Frequency branch (handles both 4D and 5D) ──
        freq_features = self.freq_branch(x)                  # (B, freq_dim)

        # ── Fusion ──
        combined = torch.cat([spatial_features, freq_features], dim=1)
        prediction = self.classifier(combined)

        return {
            "prediction": prediction,
            "spatial_features": spatial_features,
            "freq_features": freq_features,
        }

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
