"""
PGD (Projected Gradient Descent) attack.

Owner: Nitin

Algorithm:
    Random start within epsilon ball, then iterative FGSM with projection.
    Default: epsilon=4/255, steps=10, alpha=1/255

Args:
    model: deepfake detector following forward() contract
    images: (B, 3, 224, 224) clean images in [0, 1]
    labels: (B,) ground truth
    epsilon: perturbation budget
    num_steps: PGD iterations (default 10)
    alpha: step size per iteration (default epsilon/4)

Returns:
    (B, 3, 224, 224) adversarial images clamped to [0, 1]
"""

import torch
import torch.nn as nn


def pgd_attack(model, images, labels, epsilon, num_steps=10, alpha=None):
    """
    Projected Gradient Descent attack.

    Generates adversarial examples by iteratively applying FGSM-like steps
    and projecting back into the epsilon-ball around the original image.
    Uses a random start for better adversarial example diversity.
    """
    if alpha is None:
        alpha = epsilon / 4.0

    bce_loss = nn.BCELoss()
    labels_float = labels.float().unsqueeze(1)  # (B, 1)

    # Random start within epsilon ball
    delta = torch.zeros_like(images).uniform_(-epsilon, epsilon)
    delta = torch.clamp(images + delta, 0.0, 1.0) - images
    delta.requires_grad_(True)

    for _ in range(num_steps):
        adv_images = images + delta
        output = model(adv_images)
        loss = bce_loss(output["prediction"], labels_float)

        loss.backward()

        # FGSM step on delta
        grad_sign = delta.grad.data.sign()
        delta_new = delta.data + alpha * grad_sign

        # Project back into epsilon ball
        delta_new = torch.clamp(delta_new, -epsilon, epsilon)

        # Ensure adv images stay in [0, 1]
        delta_new = torch.clamp(images + delta_new, 0.0, 1.0) - images

        delta = delta_new.detach().requires_grad_(True)

    adv_images = torch.clamp(images + delta, 0.0, 1.0)
    return adv_images.detach()
