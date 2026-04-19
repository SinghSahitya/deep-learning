"""
PGD (Projected Gradient Descent) attack.

Owner: Nitin

Multi-step iterative attack with random start within the epsilon-ball.
Much stronger than FGSM; standard for adversarial robustness evaluation.
"""

import torch
import torch.nn as nn


def pgd_attack(model, images, labels, epsilon, num_steps=10, alpha=None):
    """
    PGD attack implementation.

    Args:
        model: deepfake detector following forward() contract
        images: (B, 3, 224, 224) clean images in [0, 1]
        labels: (B,) ground truth (0=real, 1=fake)
        epsilon: perturbation budget (e.g., 4/255)
        num_steps: PGD iterations (default 10)
        alpha: step size per iteration (default epsilon/4)

    Returns:
        (B, 3, 224, 224) adversarial images clamped to [0, 1]
    """
    if alpha is None:
        alpha = epsilon / 4

    was_training = model.training
    model.eval()

    criterion = nn.BCELoss()

    # Random start within epsilon-ball (distinguishes PGD from I-FGSM)
    adv_images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    adv_images = adv_images.clamp(0.0, 1.0).detach()

    for _ in range(num_steps):
        adv_images.requires_grad_(True)

        output = model(adv_images)["prediction"]
        loss = criterion(output.squeeze(1), labels.float())
        loss.backward()

        # Gradient ascent on loss
        adv_images = adv_images + alpha * adv_images.grad.sign()

        # Project back onto epsilon-ball around original
        perturbation = (adv_images - images).clamp(-epsilon, epsilon)

        # Also clamp to valid image range [0, 1]
        adv_images = (images + perturbation).clamp(0.0, 1.0).detach()

    if was_training:
        model.train()

    return adv_images
