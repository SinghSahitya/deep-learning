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
    FGSM attack implementation.

    Args:
        model: deepfake detector model
        images: (B, 3, 224, 224) clean images in [0, 1]
        labels: (B,) ground truth (0=real, 1=fake)
        epsilon: perturbation budget (e.g., 4/255)

    Returns:
        (B, 3, 224, 224) adversarial images clamped to [0, 1]
    """
    # Save and set model to eval mode
    was_training = model.training
    model.eval()

    # Clone images and enable gradient computation
    images_adv = images.clone().detach().requires_grad_(True)

    # Forward pass
    output = model(images_adv)["prediction"]  # (B, 1)

    # Compute BCE loss
    criterion = nn.BCELoss()
    loss = criterion(output.squeeze(1), labels.float())

    # Backward pass to get gradients w.r.t. input
    loss.backward()

    # Compute perturbation using the sign of the gradient
    perturbation = epsilon * images_adv.grad.sign()

    # Create adversarial images and clamp to valid range
    adversarial_images = (images + perturbation).clamp(0.0, 1.0)

    # Restore model training state
    if was_training:
        model.train()

    return adversarial_images.detach()
