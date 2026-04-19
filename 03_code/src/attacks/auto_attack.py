"""
AutoAttack evaluation wrapper.

Owner: Nitin

Wraps a sigmoid-output deepfake detector into a 2-class logit model
compatible with the AutoAttack library (Croce & Hein, 2020).
Falls back to strong PGD if autoattack is not installed.
"""

import torch
import torch.nn as nn


class ModelWrapper(nn.Module):
    """
    Wraps a sigmoid-output deepfake detector into a 2-class logit model
    for compatibility with AutoAttack.
    """

    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(self, x):
        out = self.model(x)["prediction"]  # (B, 1) sigmoid output

        fake_prob = out.squeeze(1)
        real_prob = 1.0 - fake_prob

        # Log-scale (logit-like) for AutoAttack
        logits = torch.stack(
            [
                torch.log(real_prob + 1e-8),
                torch.log(fake_prob + 1e-8),
            ],
            dim=1,
        )
        return logits


def auto_attack_eval(model, images, labels, epsilon, batch_size=32):
    """
    Run AutoAttack evaluation.

    Args:
        model: deepfake detector model
        images: (N, 3, 224, 224) test images in [0, 1]
        labels: (N,) ground truth (0=real, 1=fake)
        epsilon: perturbation budget
        batch_size: internal batch size for AutoAttack

    Returns:
        dict: {
            "adversarial_images": (N, 3, 224, 224),
            "robust_accuracy": float,
            "clean_accuracy": float,
        }
    """
    was_training = model.training
    model.eval()

    wrapped_model = ModelWrapper(model)

    with torch.no_grad():
        clean_logits = wrapped_model(images)
        clean_preds = clean_logits.argmax(dim=1)
        clean_accuracy = (clean_preds == labels).float().mean().item()

    try:
        from autoattack import AutoAttack

        adversary = AutoAttack(
            wrapped_model, norm="Linf", eps=epsilon, version="standard"
        )
        adv_images = adversary.run_standard_evaluation(
            images, labels, bs=batch_size
        )
    except ImportError:
        print("[WARNING] autoattack not installed, falling back to PGD-20")
        from .pgd import pgd_attack

        adv_images = pgd_attack(model, images, labels, epsilon, num_steps=20)

    with torch.no_grad():
        adv_logits = wrapped_model(adv_images)
        adv_preds = adv_logits.argmax(dim=1)
        robust_accuracy = (adv_preds == labels).float().mean().item()

    if was_training:
        model.train()

    return {
        "adversarial_images": adv_images,
        "robust_accuracy": robust_accuracy,
        "clean_accuracy": clean_accuracy,
    }
