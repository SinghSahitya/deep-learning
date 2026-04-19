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

import os
import torch
import torch.nn as nn
from tqdm import tqdm

from src.attacks.pgd import pgd_attack
from src.losses.combined_loss import CombinedRobustLoss


def train_adversarial(model, train_loader, val_loader, config, device):
    """
    Full adversarial training loop with AFS + frequency consistency loss.

    Implements:
    - Backbone freezing for warm-up epochs
    - On-the-fly PGD adversarial example generation
    - Combined robust loss (BCE + AFS spatial + AFS freq)
    - Cosine annealing / ReduceLROnPlateau scheduling
    - Gradient clipping to prevent explosion from adversarial training
    - Best checkpoint saving by robust (PGD) validation accuracy

    Args:
        model: MultiDomainDetector or EfficientNetDetector
        train_loader: DataLoader yielding {"image": Tensor, "label": Tensor}
        val_loader: DataLoader
        config: DotDict from YAML config
        device: 'cuda' or 'cpu'

    Returns:
        history: dict with per-epoch logs of all loss components and val metrics
    """
    # ── Extract config values ──
    epochs = config.training.epochs
    lr = config.training.lr
    weight_decay = config.training.weight_decay
    freeze_epochs = config.training.freeze_backbone_epochs
    grad_clip = config.training.get("grad_clip", 1.0)
    scheduler_type = config.training.get("scheduler", "cosine")

    lambda_afs = config.loss.lambda_afs
    lambda_freq = config.loss.lambda_freq

    eps = config.adversarial.epsilon
    pgd_steps = config.adversarial.pgd_steps
    pgd_alpha = config.adversarial.pgd_alpha
    val_pgd_steps = config.adversarial.get("val_pgd_steps", 5)

    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Setup ──
    model = model.to(device)
    criterion = CombinedRobustLoss(lambda_afs=lambda_afs, lambda_freq=lambda_freq).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
        )

    # ── History tracking ──
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_clean_acc": [],
        "val_pgd_acc": [],
        "bce_clean": [],
        "bce_adv": [],
        "afs_spatial": [],
        "afs_freq": [],
        "lr": [],
    }

    best_pgd_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ── Backbone freezing / unfreezing ──
        if epoch <= freeze_epochs:
            model.freeze_backbone()
            # Rebuild optimizer to only update unfrozen params
            if epoch == 1:
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=lr,
                    weight_decay=weight_decay,
                )
                if scheduler_type == "cosine":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=epochs, eta_min=1e-6
                    )
                else:
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
                    )
        elif epoch == freeze_epochs + 1:
            model.unfreeze_backbone()
            # Rebuild optimizer with all parameters now unfrozen
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            if scheduler_type == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs - freeze_epochs, eta_min=1e-6
                )
            else:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
                )

        # ══════════════════════════════════════════
        # TRAINING PHASE
        # ══════════════════════════════════════════
        model.train()
        epoch_loss = 0.0
        epoch_bce_clean = 0.0
        epoch_bce_adv = 0.0
        epoch_afs_spatial = 0.0
        epoch_afs_freq = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # 1. Forward pass on clean images
            clean_output = model(images)

            # 2. Generate adversarial examples using PGD
            #    Model stays in train mode for consistent BatchNorm statistics
            adv_images = pgd_attack(
                model, images, labels,
                epsilon=eps,
                num_steps=pgd_steps,
                alpha=pgd_alpha,
            )

            # 3. Detach adversarial images before second forward pass
            #    (don't backprop through the attack generation)
            adv_images = adv_images.detach()

            # 4. Forward pass on adversarial images
            adv_output = model(adv_images)

            # 5. Compute combined loss
            loss_dict = criterion(clean_output, adv_output, labels)

            # 6. Backward pass
            optimizer.zero_grad()
            loss_dict["total"].backward()

            # 7. Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            # 8. Optimizer step
            optimizer.step()

            # Accumulate for logging
            epoch_loss += loss_dict["total"].item()
            epoch_bce_clean += loss_dict["bce_clean"]
            epoch_bce_adv += loss_dict["bce_adv"]
            epoch_afs_spatial += loss_dict["afs_spatial"]
            epoch_afs_freq += loss_dict["afs_freq"]
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss_dict['total'].item():.4f}",
                "bce_c": f"{loss_dict['bce_clean']:.4f}",
                "bce_a": f"{loss_dict['bce_adv']:.4f}",
            })

        # Average training metrics
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_bce_clean = epoch_bce_clean / max(num_batches, 1)
        avg_bce_adv = epoch_bce_adv / max(num_batches, 1)
        avg_afs_spatial = epoch_afs_spatial / max(num_batches, 1)
        avg_afs_freq = epoch_afs_freq / max(num_batches, 1)

        history["train_loss"].append(avg_loss)
        history["bce_clean"].append(avg_bce_clean)
        history["bce_adv"].append(avg_bce_adv)
        history["afs_spatial"].append(avg_afs_spatial)
        history["afs_freq"].append(avg_afs_freq)

        # ══════════════════════════════════════════
        # VALIDATION PHASE
        # ══════════════════════════════════════════
        val_clean_acc, val_pgd_acc, val_loss = _validate(
            model, val_loader, criterion, device,
            eps=eps, pgd_steps=val_pgd_steps, pgd_alpha=pgd_alpha,
        )

        history["val_clean_acc"].append(val_clean_acc)
        history["val_pgd_acc"].append(val_pgd_acc)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # ── Learning rate scheduling ──
        if scheduler_type == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_pgd_acc)

        # ── Logging ──
        frozen_str = " [backbone frozen]" if epoch <= freeze_epochs else ""
        print(
            f"Epoch {epoch}/{epochs}{frozen_str} | "
            f"Loss: {avg_loss:.4f} | "
            f"BCE(c/a): {avg_bce_clean:.4f}/{avg_bce_adv:.4f} | "
            f"AFS(s/f): {avg_afs_spatial:.4f}/{avg_afs_freq:.4f} | "
            f"Val Clean: {val_clean_acc:.4f} | Val PGD: {val_pgd_acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # ── Save best checkpoint by PGD val accuracy ──
        if val_pgd_acc > best_pgd_acc:
            best_pgd_acc = val_pgd_acc
            ckpt_path = os.path.join(checkpoint_dir, "best_robust.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_clean_acc": val_clean_acc,
                "val_pgd_acc": val_pgd_acc,
                "config": dict(config),
            }, ckpt_path)
            print(f"  ✓ Saved best robust checkpoint (PGD acc: {val_pgd_acc:.4f})")

    print(f"\nTraining complete. Best PGD val accuracy: {best_pgd_acc:.4f}")
    return history


@torch.no_grad()
def _validate_clean(model, val_loader, device):
    """Compute clean validation accuracy."""
    model.eval()
    correct = 0
    total = 0

    for batch in val_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        output = model(images)
        preds = (output["prediction"].squeeze(1) >= 0.5).long()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / max(total, 1)


def _validate_pgd(model, val_loader, device, eps, pgd_steps, pgd_alpha):
    """Compute PGD validation accuracy (fewer steps for speed)."""
    model.eval()
    correct = 0
    total = 0

    for batch in val_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Generate PGD adversarial examples (model in eval mode for val)
        adv_images = pgd_attack(
            model, images, labels,
            epsilon=eps,
            num_steps=pgd_steps,
            alpha=pgd_alpha,
        )

        with torch.no_grad():
            output = model(adv_images)
            preds = (output["prediction"].squeeze(1) >= 0.5).long()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / max(total, 1)


def _validate(model, val_loader, criterion, device, eps, pgd_steps, pgd_alpha):
    """Run full validation: clean acc, PGD acc, and validation loss."""
    clean_acc = _validate_clean(model, val_loader, device)

    pgd_acc = _validate_pgd(model, val_loader, device, eps, pgd_steps, pgd_alpha)

    # Compute validation loss on clean + adv
    model.eval()
    val_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            clean_output = model(images)

        adv_images = pgd_attack(
            model, images, labels,
            epsilon=eps, num_steps=pgd_steps, alpha=pgd_alpha,
        )

        with torch.no_grad():
            adv_output = model(adv_images)
            loss_dict = criterion(clean_output, adv_output, labels)
            val_loss += loss_dict["total"].item()
            num_batches += 1

    avg_val_loss = val_loss / max(num_batches, 1)
    return clean_acc, pgd_acc, avg_val_loss
