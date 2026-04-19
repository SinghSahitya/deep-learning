import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    """
    Simple CNN baseline:
    - Conv2d(3, 32, 3, padding=1) -> BN -> ReLU -> MaxPool2d(2)
    - Conv2d(32, 64, 3, padding=1) -> BN -> ReLU -> MaxPool2d(2)
    - Conv2d(64, 128, 3, padding=1) -> BN -> ReLU -> MaxPool2d(2)
    - Conv2d(128, 256, 3, padding=1) -> BN -> ReLU -> AdaptiveAvgPool2d(1)
    - Flatten -> Linear(256, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 1) -> Sigmoid
    
    forward(x) returns:
        {
            "prediction": (B, 1),
            "spatial_features": (B, 256),  # Output of the last conv block before FC
            "freq_features": None
        }
    """
    def __init__(self):
        super(BaselineCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, 3, 224, 224)
        x = self.features(x)
        spatial_features = torch.flatten(x, 1)  # (B, 256)
        
        prediction = self.classifier(spatial_features) # (B, 1)
        
        return {
            "prediction": prediction,
            "spatial_features": spatial_features,
            "freq_features": None
        }
