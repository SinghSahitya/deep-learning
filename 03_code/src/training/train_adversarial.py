"""
Adversarial training loop with combined robust loss.

Owner: Vishesh

Per batch:
    1. Forward on clean images
    2. Generate PGD adversarial examples (BN stays in train mode)
    3. Forward on adversarial images
    4. Compute CombinedRobustLoss
    5. Backward + optimizer step

Uses adversarial warmup: ramps epsilon and PGD steps over the first
few epochs to prevent catastrophic overfitting.

Validates on both clean and PGD accuracy.
Saves best checkpoint by combined metric (0.7*clean + 0.3*pgd).
Also saves last-epoch checkpoint for resume.
"""

import os
from datetime import datetime
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.attacks.pgd import pgd_attack
from src.losses.combined_loss import CombinedRobustLoss


def _get_warmup_schedule(epoch, freeze_epochs, epochs, target_eps, target_steps):
    """Ramp adversarial strength over epochs after backbone unfreeze."""
    adv_epoch = max(0, epoch - freeze_epochs)
    total_adv_epochs = epochs - freeze_epochs
    warmup_fraction = 0.15
    warmup_epochs = max(1, int(total_adv_epochs * warmup_fraction))

    if adv_epoch <= 0:
        return target_eps * 0.5, max(2, target_steps // 2)

    if adv_epoch <= warmup_epochs:
        progress = adv_epoch / warmup_epochs
        cur_eps = target_eps * (0.5 + 0.5 * progress)
        cur_steps = max(2, int(target_steps * (0.5 + 0.5 * progress)))
        return cur_eps, cur_steps

    return target_eps, target_steps


def _get_inputs(batch, device):
    """Extract model input tensor from batch (clip or single-frame)."""
    if "clip" in batch:
        return batch["clip"].to(device, non_blocking=True)
    return batch["image"].to(device, non_blocking=True)


def train_adversarial(model, train_loader, val_loader, config, device, resume_from=None):
    """
    Full adversarial training loop with AFS + frequency consistency loss.
    Uses mixed precision (AMP) and adversarial warmup.

    Args:
        resume_from: path to checkpoint to resume training from (optional)
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

    target_eps = config.adversarial.epsilon
    target_pgd_steps = config.adversarial.pgd_steps
    pgd_alpha = config.adversarial.pgd_alpha
    val_pgd_steps = config.adversarial.get("val_pgd_steps", 7)

    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_tag = config.get("checkpoint_tag", config.model.name)
    run_id = datetime.now().strftime("%m%d_%H%M")

    # ── Setup ──
    model = model.to(device)
    criterion = CombinedRobustLoss(lambda_afs=lambda_afs, lambda_freq=lambda_freq).to(device)

    use_amp = device == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # ── Resume from checkpoint if provided ──
    start_epoch = 1
    best_combined = 0.0

    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_combined = 0.5 * ckpt.get("val_clean_acc", 0) + 0.5 * ckpt.get("val_pgd_acc", 0)
        print(f"  Resuming at epoch {start_epoch}, previous best combined: {best_combined:.4f}")

    # ── Optimizer setup ──
    # Always start with all params unfrozen if resuming past freeze phase
    if start_epoch > freeze_epochs and hasattr(model, "unfreeze_backbone"):
        model.unfreeze_backbone()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    if resume_from and os.path.exists(resume_from):
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                pass

    adv_epochs = max(1, epochs - freeze_epochs)
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=adv_epochs, T_mult=1, eta_min=1e-6
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

    for epoch in range(start_epoch, epochs + 1):
        # ── Backbone freezing / unfreezing ──
        if epoch <= freeze_epochs:
            model.freeze_backbone()
            if epoch == start_epoch:
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=lr,
                    weight_decay=weight_decay,
                )
                if scheduler_type == "cosine":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer, T_0=max(1, epochs - freeze_epochs), T_mult=1, eta_min=1e-6
                    )
                else:
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
                    )
        elif epoch == freeze_epochs + 1:
            model.unfreeze_backbone()
            backbone_lr = lr * 0.1
            head_params = []
            backbone_params = []
            for name, param in model.named_parameters():
                if "backbone" in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)
            optimizer = torch.optim.Adam([
                {"params": backbone_params, "lr": backbone_lr},
                {"params": head_params, "lr": lr},
            ], weight_decay=weight_decay)
            adv_epochs = max(1, epochs - freeze_epochs)
            if scheduler_type == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=adv_epochs, T_mult=1, eta_min=1e-6
                )
            else:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
                )

        # ── Adversarial warmup schedule ──
        cur_eps, cur_steps = _get_warmup_schedule(
            epoch, freeze_epochs, epochs, target_eps, target_pgd_steps
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
            inputs = _get_inputs(batch, device)
            labels = batch["label"].to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                clean_output = model(inputs)

            adv_inputs = pgd_attack(
                model, inputs, labels,
                epsilon=cur_eps,
                num_steps=cur_steps,
                alpha=pgd_alpha,
                use_amp=use_amp,
                keep_mode=True,
            )
            adv_inputs = adv_inputs.detach()

            with autocast(enabled=use_amp):
                adv_output = model(adv_inputs)

            loss_dict = criterion(clean_output, adv_output, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss_dict["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

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
        # VALIDATION PHASE (single pass, always at full strength)
        # ══════════════════════════════════════════
        val_clean_acc, val_pgd_acc, val_loss = _validate(
            model, val_loader, criterion, device,
            eps=target_eps, pgd_steps=val_pgd_steps, pgd_alpha=pgd_alpha,
            use_amp=use_amp,
        )

        history["val_clean_acc"].append(val_clean_acc)
        history["val_pgd_acc"].append(val_pgd_acc)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # ── Learning rate scheduling ──
        if scheduler_type == "cosine":
            scheduler.step(epoch - freeze_epochs)
        else:
            scheduler.step(val_pgd_acc)

        # ── Logging ──
        frozen_str = " [backbone frozen]" if epoch <= freeze_epochs else ""
        combined = 0.7 * val_clean_acc + 0.3 * val_pgd_acc
        print(
            f"Epoch {epoch}/{epochs}{frozen_str} | "
            f"Loss: {avg_loss:.4f} | "
            f"BCE(c/a): {avg_bce_clean:.4f}/{avg_bce_adv:.4f} | "
            f"AFS(s/f): {avg_afs_spatial:.4f}/{avg_afs_freq:.4f} | "
            f"eps: {cur_eps:.4f} steps: {cur_steps} | "
            f"Val Clean: {val_clean_acc:.4f} | Val PGD: {val_pgd_acc:.4f} | "
            f"Combined: {combined:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # ── Save best checkpoint by combined metric ──
        if combined > best_combined:
            best_combined = combined
            ckpt_path = os.path.join(checkpoint_dir, f"best_{checkpoint_tag}_{run_id}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_clean_acc": val_clean_acc,
                "val_pgd_acc": val_pgd_acc,
                "config": dict(config),
            }, ckpt_path)
            print(f"  -> Saved best checkpoint (combined: {combined:.4f}) -> {ckpt_path}")

    # ── Save last-epoch checkpoint ──
    last_path = os.path.join(checkpoint_dir, f"last_epoch_{checkpoint_tag}.pth")
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_clean_acc": val_clean_acc,
        "val_pgd_acc": val_pgd_acc,
        "config": dict(config),
    }, last_path)
    print(f"  -> Saved last-epoch checkpoint -> {last_path}")

    print(f"\nTraining complete. Best combined (0.7*clean + 0.3*pgd): {best_combined:.4f}")
    return history


def _validate(model, val_loader, criterion, device, eps, pgd_steps, pgd_alpha, use_amp=False):
    """Single-pass validation: clean acc, PGD acc, and val loss all at once."""
    model.eval()
    clean_correct = 0
    pgd_correct = 0
    total = 0
    val_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        inputs = _get_inputs(batch, device)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.no_grad(), autocast(enabled=use_amp):
            clean_output = model(inputs)
            clean_preds = (clean_output["prediction"].squeeze(1) >= 0.5).long()
            clean_correct += (clean_preds == labels).sum().item()

        adv_inputs = pgd_attack(
            model, inputs, labels,
            epsilon=eps, num_steps=pgd_steps, alpha=pgd_alpha,
            use_amp=use_amp,
        )

        with torch.no_grad(), autocast(enabled=use_amp):
            adv_output = model(adv_inputs)
            pgd_preds = (adv_output["prediction"].squeeze(1) >= 0.5).long()
            pgd_correct += (pgd_preds == labels).sum().item()

        with torch.no_grad():
            loss_dict = criterion(clean_output, adv_output, labels)
            val_loss += loss_dict["total"].item()

        total += labels.size(0)
        num_batches += 1

    clean_acc = clean_correct / max(total, 1)
    pgd_acc = pgd_correct / max(total, 1)
    avg_val_loss = val_loss / max(num_batches, 1)
    return clean_acc, pgd_acc, avg_val_loss
