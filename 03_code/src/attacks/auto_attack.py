"""
AutoAttack evaluation wrapper.

Owner: Nitin

Provides a standardised evaluation interface using the AutoAttack library
(Croce & Hein, 2020) for reliable robustness benchmarking.
Falls back to PGD if autoattack is not installed.
"""

import torch
import torch.nn as nn


def auto_attack_eval(model, images, labels, epsilon=4/255, norm='Linf', version='standard'):
    """
    Run AutoAttack evaluation on a batch of images.

    Args:
        model: deepfake detector following forward() contract
        images: (B, 3, 224, 224) clean images in [0, 1]
        labels: (B,) ground truth (0=real, 1=fake)
        epsilon: perturbation budget
        norm: 'Linf' or 'L2'
        version: 'standard' or 'plus'

    Returns:
        (B, 3, 224, 224) adversarial images
    """
    try:
        from autoattack import AutoAttack

        # Wrap model to return logits (AutoAttack expects logits, not sigmoid)
        class LogitWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                out = self.model(x)
                pred = out["prediction"]  # (B, 1) sigmoid
                # Convert sigmoid to logit: logit = log(p / (1-p))
                pred = torch.clamp(pred, 1e-7, 1 - 1e-7)
                logits = torch.log(pred / (1 - pred))
                # AutoAttack expects (B, num_classes) — binary: stack [1-p, p]
                return torch.cat([1 - pred, pred], dim=1)

        wrapper = LogitWrapper(model)
        adversary = AutoAttack(wrapper, norm=norm, eps=epsilon, version=version, verbose=False)
        adv_images = adversary.run_standard_evaluation(images, labels, bs=len(images))
        return adv_images

    except ImportError:
        # Fallback to PGD if autoattack is not installed
        from .pgd import pgd_attack
        print("[WARNING] autoattack not installed, falling back to PGD-20")
        return pgd_attack(model, images, labels, epsilon, num_steps=20)
