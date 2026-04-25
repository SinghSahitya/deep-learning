"""
Frequency-domain feature extraction branch with temporal 3D FFT.

Owner: Vishesh

Clip pipeline (T > 1):
    (B, T, 3, H, W) -> Grayscale -> (B, 1, T, H, W)
    -> 3D FFT over (T, H, W) -> fftshift -> log(1 + |spectrum|)
    -> 3D CNN -> AdaptiveAvgPool3d(1) -> (B, output_dim)

    Captures temporal flickering / inter-frame inconsistencies AND
    spatial compression artifacts simultaneously.

Single-frame fallback (T == 1 or 4-D input):
    Same as original 2D FFT -> 2D CNN path.
"""

import torch
import torch.nn as nn


class FrequencyBranch(nn.Module):
    """
    Extracts features from the spatio-temporal frequency domain of video clips.

    When given a clip of T frames, applies a 3D FFT across (T, H, W) to
    capture temporal frequency features (flickering, frame-to-frame
    inconsistencies) alongside spatial frequency features (compression
    artifacts, blending boundaries).

    Falls back to 2D FFT for single-frame inputs (backward compatible).

    The FFT pipeline is fully differentiable (torch.fft supports autograd).
    """

    def __init__(self, output_dim=128):
        super().__init__()
        self.output_dim = output_dim

        self.cnn3d = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(64, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(output_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )

        self.cnn2d = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def _to_gray(self, rgb):
        """Standard luminance weighting, works on any trailing spatial dims."""
        return 0.299 * rgb[..., 0:1, :, :] + 0.587 * rgb[..., 1:2, :, :] + 0.114 * rgb[..., 2:3, :, :]

    def forward(self, x):
        """
        Args:
            x: (B, T, 3, H, W) clip  OR  (B, 3, H, W) single frame
        Returns:
            (B, output_dim) frequency feature vector
        """
        if x.dim() == 4:
            return self._forward_2d(x)
        return self._forward_3d(x)

    def _forward_3d(self, x):
        """3D FFT path for clips: captures temporal + spatial frequency."""
        B, T, C, H, W = x.shape
        gray = self._to_gray(x)            # (B, T, 1, H, W)
        gray = gray.squeeze(2)              # (B, T, H, W)
        gray = gray.unsqueeze(1)            # (B, 1, T, H, W)

        spectrum = torch.fft.fftn(gray, dim=(2, 3, 4))
        spectrum = torch.fft.fftshift(spectrum, dim=(2, 3, 4))
        magnitude = torch.log(1 + torch.abs(spectrum))

        features = self.cnn3d(magnitude)    # (B, output_dim, 1, 1, 1)
        return features.flatten(1)          # (B, output_dim)

    def _forward_2d(self, x):
        """2D FFT fallback for single frames (backward compatible)."""
        gray = self._to_gray(x)             # (B, 1, H, W)

        spectrum = torch.fft.fft2(gray)
        spectrum = torch.fft.fftshift(spectrum)
        magnitude = torch.log(1 + torch.abs(spectrum))

        features = self.cnn2d(magnitude)    # (B, output_dim, 1, 1)
        return features.flatten(1)          # (B, output_dim)
