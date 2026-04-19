"""
Evaluation framework: clean + adversarial robustness testing.

Owner: Nitin

evaluate_model: Test model under clean data and specified attacks.
run_full_evaluation: Convenience wrapper with all standard attacks.
"""

import torch
import numpy as np
from tqdm import tqdm

from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.utils.metrics import compute_metrics


def evaluate_model(model, test_loader, device, attacks=None):
    """
    Evaluate model on clean data and under adversarial attacks.

    Args:
        model: detector model
        test_loader: DataLoader for test set
        device: 'cuda' or 'cpu'
        attacks: dict of attack_name -> attack_function
                 e.g., {"fgsm_2": lambda m,x,y: fgsm_attack(m,x,y,2/255),
                        "pgd_4": lambda m,x,y: pgd_attack(m,x,y,4/255)}

    Returns:
        dict: {condition_name: {"accuracy", "auc", "predictions", "labels", ...}}
    """
    if attacks is None:
        attacks = {}

    model.eval()
    model.to(device)

    # Collect all conditions to evaluate
    conditions = ["clean"] + list(attacks.keys())
    all_predictions = {c: [] for c in conditions}
    all_labels = {c: [] for c in conditions}

    with torch.no_grad() if len(attacks) == 0 else torch.enable_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # --- Clean evaluation ---
            with torch.no_grad():
                clean_output = model(images)["prediction"]  # (B, 1)
            clean_preds = clean_output.squeeze(1).cpu().numpy()
            all_predictions["clean"].extend(clean_preds.tolist())
            all_labels["clean"].extend(labels.cpu().numpy().tolist())

            # --- Adversarial evaluations ---
            for attack_name, attack_fn in attacks.items():
                # Generate adversarial images (needs gradients)
                adv_images = attack_fn(model, images, labels)

                # Evaluate on adversarial images
                with torch.no_grad():
                    adv_output = model(adv_images)["prediction"]  # (B, 1)
                adv_preds = adv_output.squeeze(1).cpu().numpy()
                all_predictions[attack_name].extend(adv_preds.tolist())
                all_labels[attack_name].extend(labels.cpu().numpy().tolist())

    # Compute metrics for each condition
    results = {}
    for condition in conditions:
        metrics = compute_metrics(
            all_predictions[condition], all_labels[condition]
        )
        metrics["predictions"] = all_predictions[condition]
        metrics["labels"] = all_labels[condition]
        results[condition] = metrics

    return results


def run_full_evaluation(model, test_loader, device):
    """
    Convenience function that runs all standard attacks.
    Defines the attack dict with FGSM (eps=2,4,8/255), PGD (eps=4/255),
    and calls evaluate_model.

    Returns:
        results dict from evaluate_model
    """
    attacks = {
        "fgsm_eps2": lambda m, x, y: fgsm_attack(m, x, y, epsilon=2 / 255),
        "fgsm_eps4": lambda m, x, y: fgsm_attack(m, x, y, epsilon=4 / 255),
        "fgsm_eps8": lambda m, x, y: fgsm_attack(m, x, y, epsilon=8 / 255),
        "pgd_eps4": lambda m, x, y: pgd_attack(
            m, x, y, epsilon=4 / 255, num_steps=10, alpha=1 / 255
        ),
    }
    return evaluate_model(model, test_loader, device, attacks=attacks)
