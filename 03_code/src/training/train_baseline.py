"""
Baseline training loop (clean data, BCE loss only).

Owner: Sahitya

- Adam optimizer with LR and weight_decay from config
- BCELoss
- Backbone freezing for first N epochs
- Saves best checkpoint by val accuracy
- Returns training history dict
"""

import torch
import torch.nn as nn


def train_baseline(model, train_loader, val_loader, config, device):
    """
    Returns:
        history: {"train_loss": [...], "val_loss": [...], "val_acc": [...]}
    """
    # TODO
    raise NotImplementedError
