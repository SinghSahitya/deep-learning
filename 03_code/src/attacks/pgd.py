"""
PGD (Projected Gradient Descent) attack.

Owner: Nitin

Multi-step iterative attack with random start within the epsilon-ball.
Much stronger than FGSM; standard for adversarial robustness evaluation.
"""

import torch
import torch.nn as nn


def pgd_attack(model, images, labels, epsilon, num_steps=10, alpha=None, use_amp=False, keep_mode=False):
    """
    PGD attack implementation.

    Args:
        model: deepfake detector following forward() contract
        images: clean images in [0, 1] (any shape the model accepts)
        labels: (B,) ground truth (0=real, 1=fake)
        epsilon: perturbation budget (e.g., 4/255)
        num_steps: PGD iterations (default 10)
        alpha: step size per iteration (default epsilon/4)
        use_amp: ignored (kept for API compat, PGD always runs in float32)
        keep_mode: if True, don't switch to eval (used during adversarial training
                   to keep BN in train mode for consistency)

    Returns:
        adversarial images clamped to [0, 1], same shape as input
    """
    if alpha is None:
        alpha = epsilon / 4

    was_training = model.training
    if not keep_mode:
        model.eval()

    criterion = nn.BCELoss()

    adv_images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    adv_images = adv_images.clamp(0.0, 1.0).detach()

    for _ in range(num_steps):
        adv_images.requires_grad_(True)

        with torch.cuda.amp.autocast(enabled=False):
            output = model(adv_images.float())["prediction"]
            pred = output.squeeze(1).clamp(1e-7, 1 - 1e-7)
            loss = criterion(pred, labels.float())

        loss.backward()

        adv_images = adv_images + alpha * adv_images.grad.sign()
        perturbation = (adv_images - images).clamp(-epsilon, epsilon)
        adv_images = (images + perturbation).clamp(0.0, 1.0).detach()

    if not keep_mode and was_training:
        model.train()

    return adv_images
