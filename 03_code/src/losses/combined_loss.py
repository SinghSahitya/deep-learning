"""
Combined robust loss for adversarial training.

Owner: Vishesh

L_total = L_BCE(clean) + L_BCE(adv) + lambda_afs * L_AFS_spatial + lambda_freq * L_AFS_freq

forward() returns dict:
    {"total": loss, "bce_clean": float, "bce_adv": float, "afs_spatial": float, "afs_freq": float}
"""

import torch
import torch.nn as nn
from .adversarial_feature_similarity import AdversarialFeatureSimilarityLoss


class CombinedRobustLoss(nn.Module):
    def __init__(self, lambda_afs=0.5, lambda_freq=0.3):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, clean_output, adv_output, labels):
        """
        Args:
            clean_output: dict from model(clean_images)
            adv_output: dict from model(adv_images)
            labels: (B,) ground truth
        """
        # TODO
        raise NotImplementedError
