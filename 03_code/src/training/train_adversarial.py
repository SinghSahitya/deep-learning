"""
Adversarial training loop with combined robust loss.

Owner: Vishesh

Per batch:
    1. Forward on clean images
    2. Generate PGD adversarial examples
    3. Forward on adversarial images
    4. Compute CombinedRobustLoss
    5. Backward + optimizer step

Validates on both clean and PGD accuracy.
Saves best checkpoint by robust (PGD) val accuracy.
"""

import torch
from src.attacks.pgd import pgd_attack
from src.losses.combined_loss import CombinedRobustLoss


def train_adversarial(model, train_loader, val_loader, config, device):
    """
    Returns:
        history: {"train_loss": [...], "val_loss": [...], "val_clean_acc": [...], "val_pgd_acc": [...],
                  "bce_clean": [...], "bce_adv": [...], "afs_spatial": [...], "afs_freq": [...]}
    """
    # TODO
    raise NotImplementedError
