"""
Adversarial Feature Similarity (AFS) Loss.

Owner: Vishesh

From Khan et al. (2024): "Adversarially Robust Deepfake Detection via
Adversarial Feature Similarity Learning"

L_AFS = mean(||f(x_clean) - f(x_adv)||_2)

Forces the model to produce similar features for clean and adversarial inputs.
"""

import torch
import torch.nn as nn


class AdversarialFeatureSimilarityLoss(nn.Module):
    def forward(self, clean_features, adv_features):
        """
        Args:
            clean_features: (B, D) features from clean images
            adv_features: (B, D) features from adversarial images
        Returns:
            Scalar loss
        """
        # TODO
        raise NotImplementedError
