"""
FGSM (Fast Gradient Sign Method) attack.

Owner: Nitin

Algorithm:
    x_adv = x + epsilon * sign(grad_x(BCE_loss))

Args:
    model: deepfake detector following forward() contract
    images: (B, 3, 224, 224) clean images in [0, 1]
    labels: (B,) ground truth (0=real, 1=fake)
    epsilon: perturbation budget (e.g., 4/255)

Returns:
    (B, 3, 224, 224) adversarial images clamped to [0, 1]
"""

import torch
import torch.nn as nn


def fgsm_attack(model, images, labels, epsilon):
    """
    Fast Gradient Sign Method — single-step adversarial attack.

    Computes the gradient of the BCE loss w.r.t. the input images,
    then perturbs in the sign direction scaled by epsilon.
    """
    bce_loss = nn.BCELoss()
    labels_float = labels.float().unsqueeze(1)  # (B, 1)

    images_adv = images.clone().detach().requires_grad_(True)

    output = model(images_adv)
    loss = bce_loss(output["prediction"], labels_float)
    loss.backward()

    # Perturb in the gradient sign direction
    grad_sign = images_adv.grad.data.sign()
    adv_images = images_adv.data + epsilon * grad_sign

    # Clamp to valid image range
    adv_images = torch.clamp(adv_images, 0.0, 1.0)

    return adv_images.detach()
