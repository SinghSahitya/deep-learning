import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json

def train_baseline(model, train_loader, val_loader, config, device, save_dir="05_results/models/"):
    """
    Standard training loop for baseline models.
    """
    epochs = config['training'].get('epochs', 20)
    lr = config['training'].get('lr', 1e-4)
    weight_decay = config['training'].get('weight_decay', 1e-5)
    freeze_epochs = config['training'].get('freeze_backbone_epochs', 5)
    
    os.makedirs(save_dir, exist_ok=True)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Backbone freezing logic (if applicable, like for EfficientNet)
        if hasattr(model, 'backbone'):
            if epoch < freeze_epochs:
                for param in model.backbone.parameters():
                    param.requires_grad = False
                print("Backbone frozen.")
            elif epoch == freeze_epochs:
                for param in model.backbone.parameters():
                    param.requires_grad = True
                print("Backbone unfrozen for fine-tuning.")
                # Optionally reduce learning rate for fine-tuning
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * 0.1
        
        # Train Loop
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc="Training"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            predictions = outputs['prediction']
            
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        corrects = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images)
                predictions = outputs['prediction']
                
                loss = criterion(predictions, labels)
                val_loss += loss.item()
                
                # Accuracy calc (Predictions are probabilities from Sigmoid)
                preds_binary = (predictions > 0.5).float()
                corrects += (preds_binary == labels).sum().item()
                total += labels.size(0)
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = corrects / total
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(save_dir, f"best_{config['model']['name']}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model with Val Acc: {best_val_acc:.4f}")
            
    # Save training history
    history_path = os.path.join(save_dir, f"history_{config['model']['name']}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
        
    return history
