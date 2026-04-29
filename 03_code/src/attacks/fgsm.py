"""
FGSM (Fast Gradient Sign Method) attack.

Owner: Nitin

Single-step adversarial attack. Computes gradient of BCE loss w.r.t. input,
perturbs in sign direction scaled by epsilon.
"""

import torch
import torch.nn as nn


def fgsm_attack(model, images, labels, epsilon, use_amp=False):
    """
    FGSM attack implementation.

    Args:
        model: deepfake detector following forward() contract
        images: clean images in [0, 1] (any shape the model accepts)
        labels: (B,) ground truth (0=real, 1=fake)
        epsilon: perturbation budget (e.g., 4/255)
        use_amp: ignored (kept for API compat, attack always runs in float32)

    Returns:
        adversarial images clamped to [0, 1], same shape as input
    """
    was_training = model.training
    model.eval()

    images_adv = images.clone().detach().requires_grad_(True)

    with torch.cuda.amp.autocast(enabled=False):
        output = model(images_adv.float())["prediction"]
        pred = output.squeeze(1).clamp(1e-7, 1 - 1e-7)

    criterion = nn.BCELoss()
    loss = criterion(pred, labels.float())
    loss.backward()

    perturbation = epsilon * images_adv.grad.sign()
    adversarial_images = (images + perturbation).clamp(0.0, 1.0)

    if was_training:
        model.train()

    return adversarial_images.detach()
