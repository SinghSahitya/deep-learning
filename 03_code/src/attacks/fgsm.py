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
    # TODO
    raise NotImplementedError
