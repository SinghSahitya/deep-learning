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
    """
    AFS Loss: Minimizes the L2 distance between features extracted from
    clean and adversarial versions of the same input.

    This encourages the model to learn representations that are INVARIANT
    to adversarial perturbations — small input changes should not cause
    large feature-space changes.
    """

    def forward(self, clean_features, adv_features):
        """
        Args:
            clean_features: (B, D) features from clean images
            adv_features: (B, D) features from adversarial images
        Returns:
            Scalar loss value (mean L2 distance across the batch)
        """
        # Per-sample L2 distance: ||f(x_i) - f(x_i_adv)||_2
        distances = torch.norm(clean_features - adv_features, p=2, dim=1)  # (B,)

        # Return mean across batch
        return distances.mean()
