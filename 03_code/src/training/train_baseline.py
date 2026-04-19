"""
Baseline training loop (clean data, BCE loss only).

Owner: Sahitya

- Adam optimizer with LR and weight_decay from config
- BCELoss
- Backbone freezing for first N epochs
- Saves best checkpoint by val accuracy
- Returns training history dict
"""

import os
import torch
import torch.nn as nn
from tqdm import tqdm


def train_baseline(model, train_loader, val_loader, config, device):
    """
    Standard (non-adversarial) training loop with BCE loss.

    Args:
        model: BaselineCNN, EfficientNetDetector, or MultiDomainDetector
        train_loader: DataLoader yielding {"image": Tensor, "label": Tensor}
        val_loader: DataLoader
        config: DotDict from YAML config
        device: 'cuda' or 'cpu'

    Returns:
        history: {"train_loss": [...], "val_loss": [...], "val_acc": [...]}
    """
    epochs = config.training.epochs
    lr = config.training.lr
    weight_decay = config.training.weight_decay
    freeze_epochs = config.training.get("freeze_backbone_epochs", 0)

    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = model.to(device)
    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # Backbone freezing
        if hasattr(model, "freeze_backbone"):
            if epoch <= freeze_epochs:
                model.freeze_backbone()
            elif epoch == freeze_epochs + 1:
                model.unfreeze_backbone()
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=lr, weight_decay=weight_decay
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs - freeze_epochs, eta_min=1e-6
                )

        # ── Train ──
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].float().unsqueeze(1).to(device)

            output = model(images)
            loss = criterion(output["prediction"], labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_train_loss = epoch_loss / max(num_batches, 1)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                labels_float = labels.float().unsqueeze(1)

                output = model(images)
                loss = criterion(output["prediction"], labels_float)
                val_loss += loss.item()

                preds = (output["prediction"].squeeze(1) >= 0.5).long()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / max(num_batches, 1)
        val_acc = correct / max(total, 1)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
            }, os.path.join(checkpoint_dir, "best_baseline.pth"))
            print(f"  ✓ Saved best checkpoint (acc: {val_acc:.4f})")

    return history
