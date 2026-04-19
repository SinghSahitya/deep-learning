import os
import argparse
import torch

# Add src to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import load_config
from src.data.dataset import get_dataloaders
from src.models.baseline_cnn import BaselineCNN
from src.models.efficientnet_detector import EfficientNetDetector
from src.training.train_baseline import train_baseline

def parse_args():
    parser = argparse.ArgumentParser(description="Train Deepfake Detector")
    parser.add_argument("--config", type=str, default="03_code/configs/baseline.yaml", help="Path to config YAML")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(config)
    print(f"Batches per epoch - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    print("Initializing model...")
    model_name = config['model'].get('name', 'efficientnet')
    
    if model_name == 'baseline_cnn':
        model = BaselineCNN().to(device)
    elif model_name == 'efficientnet':
        model = EfficientNetDetector().to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    print(f"Model {model_name} initialized.")
    
    print("Starting training...")
    # NOTE: Calling train_baseline for now. Person C will add train_adversarial later.
    train_baseline(model, train_loader, val_loader, config, device)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
