"""
Combined robust loss for adversarial training.

Owner: Vishesh

L_total = L_BCE(clean) + L_BCE(adv) + lambda_afs * L_AFS_spatial + lambda_freq * L_AFS_freq

forward() returns dict:
    {"total": loss, "bce_clean": float, "bce_adv": float, "afs_spatial": float, "afs_freq": float}
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from .adversarial_feature_similarity import AdversarialFeatureSimilarityLoss


class CombinedRobustLoss(nn.Module):
    """
    Combined loss for adversarially robust deepfake detection.

    Combines:
    - BCE on clean predictions (standard classification)
    - BCE on adversarial predictions (adversarial classification)
    - Spatial AFS loss (from Khan et al. 2024)
    - Frequency AFS loss (OUR CONTRIBUTION — extends AFS to freq features)

    The frequency consistency loss applied to a dedicated frequency branch
    is the novel contribution that distinguishes this work from Khan et al.
    """

    def __init__(self, lambda_afs=0.5, lambda_freq=0.3):
        super().__init__()
        self.bce = nn.BCELoss()
        self.afs = AdversarialFeatureSimilarityLoss()
        self.lambda_afs = lambda_afs
        self.lambda_freq = lambda_freq

    @autocast(enabled=False)
    def forward(self, clean_output, adv_output, labels):
        """
        Args:
            clean_output: dict from model(clean_images)
                {"prediction": (B,1), "spatial_features": (B,256), "freq_features": (B,128)}
            adv_output: dict from model(adv_images)
                {"prediction": (B,1), "spatial_features": (B,256), "freq_features": (B,128)}
            labels: (B,) ground truth labels (0=real, 1=fake)
        Returns:
            dict with "total" (Tensor for backprop) and component values (floats for logging)
        """
        labels_float = labels.float().unsqueeze(1)  # (B, 1)

        # ── Classification loss on clean and adversarial predictions ──
        bce_clean = self.bce(clean_output["prediction"].float(), labels_float)
        bce_adv = self.bce(adv_output["prediction"].float(), labels_float)

        # ── Spatial AFS loss (from Khan et al.) ──
        afs_spatial = self.afs(
            clean_output["spatial_features"].float(),
            adv_output["spatial_features"].float()
        )

        # ── Frequency consistency loss (OUR CONTRIBUTION) ──
        # If freq_features is None (e.g., spatial-only model), skip freq loss
        if clean_output["freq_features"] is not None and adv_output["freq_features"] is not None:
            afs_freq = self.afs(
                clean_output["freq_features"].float(),
                adv_output["freq_features"].float()
            )
        else:
            afs_freq = torch.tensor(0.0, device=labels.device)

        # ── Total loss ──
        total = (
            bce_clean
            + bce_adv
            + self.lambda_afs * afs_spatial
            + self.lambda_freq * afs_freq
        )

        return {
            "total": total,
            "bce_clean": bce_clean.item(),
            "bce_adv": bce_adv.item(),
            "afs_spatial": afs_spatial.item(),
            "afs_freq": afs_freq.item(),
        }
