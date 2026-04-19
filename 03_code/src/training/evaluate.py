"""
Evaluation framework: clean + adversarial robustness testing.

Owner: Nitin

evaluate_model: Test model under clean data and specified attacks.
run_full_evaluation: Convenience wrapper with all standard attacks.
"""

import torch
from functools import partial
from tqdm import tqdm

from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.utils.metrics import compute_metrics


def evaluate_model(model, test_loader, device, attacks=None):
    """
    Evaluate model on clean data and under adversarial attacks.

    Args:
        model: deepfake detector following forward() contract
        test_loader: DataLoader yielding {"image": Tensor, "label": Tensor}
        device: 'cuda' or 'cpu'
        attacks: dict of {name: attack_fn(model, images, labels)} or None

    Returns:
        dict: {condition_name: {"accuracy": float, "auc": float, ...}}
    """
    model.eval()
    model = model.to(device)

    conditions = {"clean": None}
    if attacks:
        conditions.update(attacks)

    results = {}

    for cond_name, attack_fn in conditions.items():
        all_preds = []
        all_labels = []

        pbar = tqdm(test_loader, desc=f"Evaluating [{cond_name}]", leave=False)
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            if attack_fn is not None:
                images = attack_fn(model, images, labels)

            with torch.no_grad():
                output = model(images)
                preds = output["prediction"].squeeze(1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().tolist())

        metrics = compute_metrics(all_preds, all_labels)
        metrics["predictions"] = all_preds
        metrics["labels"] = all_labels
        results[cond_name] = metrics

        print(f"  {cond_name}: Acc={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")

    return results


def run_full_evaluation(model, test_loader, device):
    """
    Run comprehensive evaluation: clean + FGSM(2,4,8/255) + PGD(4/255).

    Args:
        model: deepfake detector
        test_loader: DataLoader
        device: 'cuda' or 'cpu'

    Returns:
        dict of evaluation results per condition
    """
    attacks = {
        "fgsm_2": partial(fgsm_attack, epsilon=2 / 255),
        "fgsm_4": partial(fgsm_attack, epsilon=4 / 255),
        "fgsm_8": partial(fgsm_attack, epsilon=8 / 255),
        "pgd_4": partial(pgd_attack, epsilon=4 / 255, num_steps=10, alpha=1 / 255),
    }

    return evaluate_model(model, test_loader, device, attacks=attacks)
