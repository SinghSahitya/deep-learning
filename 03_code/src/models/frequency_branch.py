"""
Frequency-domain feature extraction branch.

Owner: Vishesh

Pipeline:
    RGB -> Grayscale -> FFT2D -> fftshift -> log(1 + |spectrum|) -> CNN -> (B, 128)

CNN layers:
    Conv2d(1,32,3,pad=1)->BN->ReLU->MaxPool
    Conv2d(32,64,3,pad=1)->BN->ReLU->MaxPool
    Conv2d(64,128,3,pad=1)->BN->ReLU->AdaptiveAvgPool(1)
    Flatten -> (B, 128)
"""

import torch
import torch.nn as nn


class FrequencyBranch(nn.Module):
    """
    Extracts features from the frequency domain of input images.

    The FFT pipeline is fully differentiable (torch.fft supports autograd),
    enabling end-to-end training with adversarial loss backpropagation.
    """

    def __init__(self, output_dim=128):
        super().__init__()
        self.output_dim = output_dim

        self.cnn = nn.Sequential(
            # Block 1: (B, 1, 224, 224) -> (B, 32, 112, 112)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: (B, 32, 112, 112) -> (B, 64, 56, 56)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: (B, 64, 56, 56) -> (B, 128, 1, 1)
            nn.Conv2d(64, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) RGB images in [0, 1]
        Returns:
            (B, output_dim) frequency feature vector
        """
        # Step 1: RGB to grayscale using standard luminance weights
        gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        # gray shape: (B, 1, 224, 224)

        # Step 2: 2D FFT
        spectrum = torch.fft.fft2(gray)

        # Step 3: Shift zero-frequency component to center
        spectrum = torch.fft.fftshift(spectrum)

        # Step 4: Log magnitude spectrum (add 1 for numerical stability)
        magnitude = torch.log(1 + torch.abs(spectrum))
        # magnitude shape: (B, 1, 224, 224)

        # Step 5: Extract features via CNN
        features = self.cnn(magnitude)  # (B, output_dim, 1, 1)
        features = features.flatten(1)  # (B, output_dim)

        return features
