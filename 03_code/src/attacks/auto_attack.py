"""
AutoAttack wrapper for standardized robustness evaluation.

Owner: Nitin

Uses the `autoattack` library (pip install autoattack).
Wraps our sigmoid-output model into a 2-class logit model for compatibility.
Runs the standard evaluation (APGD-CE, APGD-DLR, FAB, Square).

NOTE: AutoAttack is slow. Run on a subset (500-1000 samples).
"""

import torch
import torch.nn as nn


def auto_attack_eval(model, images, labels, epsilon, batch_size=32):
    """
    Returns:
        dict: {"adversarial_images": Tensor, "robust_accuracy": float, "clean_accuracy": float}
    """
    # TODO: Wrap model to output 2-class logits, run AutoAttack
    raise NotImplementedError
