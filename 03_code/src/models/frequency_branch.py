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
    def __init__(self, output_dim=128):
        super().__init__()
        # TODO: Build CNN for frequency magnitude spectrum
        raise NotImplementedError

    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) RGB images in [0, 1]
        Returns:
            (B, output_dim) frequency feature vector
        """
        # TODO: grayscale -> fft2 -> fftshift -> log magnitude -> CNN
        raise NotImplementedError
