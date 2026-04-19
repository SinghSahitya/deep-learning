"""
FGSM (Fast Gradient Sign Method) attack.

Owner: Nitin

Single-step adversarial attack. Computes gradient of BCE loss w.r.t. input,
perturbs in sign direction scaled by epsilon.
"""

import torch
import torch.nn as nn


def fgsm_attack(model, images, labels, epsilon):
    """
    FGSM attack implementation.

    Args:
        model: deepfake detector following forward() contract
        images: (B, 3, 224, 224) clean images in [0, 1]
        labels: (B,) ground truth (0=real, 1=fake)
        epsilon: perturbation budget (e.g., 4/255)

    Returns:
        (B, 3, 224, 224) adversarial images clamped to [0, 1]
    """
    was_training = model.training
    model.eval()

    images_adv = images.clone().detach().requires_grad_(True)

    output = model(images_adv)["prediction"]  # (B, 1)

    criterion = nn.BCELoss()
    loss = criterion(output.squeeze(1), labels.float())

    loss.backward()

    perturbation = epsilon * images_adv.grad.sign()
    adversarial_images = (images + perturbation).clamp(0.0, 1.0)

    if was_training:
        model.train()

    return adversarial_images.detach()
