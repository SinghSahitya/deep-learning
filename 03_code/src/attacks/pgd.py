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
    # TODO
    raise NotImplementedError
