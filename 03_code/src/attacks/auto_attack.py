"""
AutoAttack wrapper for standardized robustness evaluation.

Owner: Nitin

Uses the `autoattack` library (pip install autoattack).
Wraps our sigmoid-output model into a 2-class logit model for compatibility.
Runs the standard evaluation (APGD-CE, APGD-DLR, FAB, Square).

NOTE: AutoAttack is slow. Run on a subset (500-1000 samples).
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

        # Convert sigmoid output to 2-class logits for AutoAttack
        fake_prob = out.squeeze(1)  # (B,) probability of fake
        real_prob = 1.0 - fake_prob  # (B,) probability of real

        # Convert to log-scale (logit-like) for AutoAttack
        logits = torch.stack(
            [
                torch.log(real_prob + 1e-8),
                torch.log(fake_prob + 1e-8),
            ],
            dim=1,
        )  # (B, 2)
        return logits


def auto_attack_eval(model, images, labels, epsilon, batch_size=32):
    """
    Run AutoAttack evaluation.

    Args:
        model: deepfake detector model
        images: (N, 3, 224, 224) ALL test images (not just one batch)
        labels: (N,) ground truth
        epsilon: perturbation budget
        batch_size: internal batch size for AutoAttack

    Returns:
        dict: {"adversarial_images": Tensor, "robust_accuracy": float, "clean_accuracy": float}
    """
    device = next(model.parameters()).device

    # Lazy import — autoattack is an optional heavy dependency
    from autoattack import AutoAttack

    # Save and set model to eval mode
    was_training = model.training
    model.eval()

    # Compute clean accuracy first
    wrapped_model = ModelWrapper(model)
    with torch.no_grad():
        clean_logits = wrapped_model(images)
        clean_preds = clean_logits.argmax(dim=1)
        clean_accuracy = (clean_preds == labels).float().mean().item()

    # Run AutoAttack (standard evaluation: APGD-CE, APGD-DLR, FAB, Square)
    adversary = AutoAttack(
        wrapped_model, norm="Linf", eps=epsilon, version="standard"
    )

    adv_images = adversary.run_standard_evaluation(
        images, labels, bs=batch_size
    )

    # Compute robust accuracy on adversarial images
    with torch.no_grad():
        adv_logits = wrapped_model(adv_images)
        adv_preds = adv_logits.argmax(dim=1)
        robust_accuracy = (adv_preds == labels).float().mean().item()

    # Restore model training state
    if was_training:
        model.train()

    return {
        "adversarial_images": adv_images,
        "robust_accuracy": robust_accuracy,
        "clean_accuracy": clean_accuracy,
    }
