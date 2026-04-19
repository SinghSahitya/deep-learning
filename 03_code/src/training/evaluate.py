"""
Evaluation framework: clean + adversarial robustness testing.

Owner: Nitin

evaluate_model: Test model under clean data and specified attacks.
run_full_evaluation: Convenience wrapper with all standard attacks.
"""

import torch


def evaluate_model(model, test_loader, device, attacks=None):
    """
    Args:
        attacks: dict of name -> attack_fn(model, images, labels)
    Returns:
        dict: {condition_name: {"accuracy": float, "auc": float, "predictions": list, "labels": list}}
    """
    # TODO
    raise NotImplementedError


def run_full_evaluation(model, test_loader, device):
    """Run clean + FGSM(2,4,8/255) + PGD(4/255) evaluation."""
    # TODO
    raise NotImplementedError
