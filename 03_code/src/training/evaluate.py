"""
Evaluation framework: clean + adversarial robustness testing.

Owner: Nitin

evaluate_model: Test model under clean data and specified attacks.
run_full_evaluation: Convenience wrapper with all standard attacks.
"""

import torch
from tqdm import tqdm

from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.utils.metrics import compute_metrics


def _get_inputs(batch, device):
    """Extract model input tensor from batch (clip or single-frame)."""
    if "clip" in batch:
        return batch["clip"].to(device)
    return batch["image"].to(device)


def evaluate_model(model, test_loader, device, attacks=None):
    """
    Evaluate model on clean data and under adversarial attacks.

    Args:
        model: detector model
        test_loader: DataLoader for test set (clip or single-frame)
        device: 'cuda' or 'cpu'
        attacks: dict of attack_name -> attack_function

    Returns:
        dict: {condition_name: {"accuracy", "auc", "predictions", "labels", ...}}
    """
    if attacks is None:
        attacks = {}

    model.eval()
    model.to(device)

    conditions = ["clean"] + list(attacks.keys())
    all_predictions = {c: [] for c in conditions}
    all_labels = {c: [] for c in conditions}

    outer_ctx = torch.enable_grad() if len(attacks) > 0 else torch.no_grad()

    with outer_ctx:
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs = _get_inputs(batch, device)
            labels = batch["label"].to(device)

            with torch.no_grad():
                clean_output = model(inputs)["prediction"]
            clean_preds = clean_output.squeeze(1).cpu().numpy()
            all_predictions["clean"].extend(clean_preds.tolist())
            all_labels["clean"].extend(labels.cpu().numpy().tolist())

            for attack_name, attack_fn in attacks.items():
                adv_inputs = attack_fn(model, inputs, labels)
                with torch.no_grad():
                    adv_output = model(adv_inputs)["prediction"]
                adv_preds = adv_output.squeeze(1).cpu().numpy()
                all_predictions[attack_name].extend(adv_preds.tolist())
                all_labels[attack_name].extend(labels.cpu().numpy().tolist())

    results = {}
    for condition in conditions:
        metrics = compute_metrics(all_predictions[condition], all_labels[condition])
        metrics["predictions"] = all_predictions[condition]
        metrics["labels"] = all_labels[condition]
        results[condition] = metrics
        print(
            f"  {condition:25s}  Acc: {metrics['accuracy']:.4f}  "
            f"AUC: {metrics['auc']:.4f}"
        )

    return results


def run_full_evaluation(model, test_loader, device):
    """
    Run comprehensive evaluation: clean + FGSM(2,4,8/255) + PGD(4/255).

    Returns dict of evaluation results per condition.
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
