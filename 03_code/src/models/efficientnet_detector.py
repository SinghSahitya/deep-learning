import torch
import torch.nn as nn
import timm

class EfficientNetDetector(nn.Module):
    """
    EfficientNet-B4 backbone with custom classification head.
    
    __init__:
        - self.backbone = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
          # num_classes=0 removes the original head, outputs (B, 1792)
        - self.feature_fc = nn.Linear(1792, 256)  # Project to 256-d
        - self.classifier = nn.Sequential(
              nn.Linear(256, 128),
              nn.ReLU(),
              nn.Dropout(0.3),
              nn.Linear(128, 1),
              nn.Sigmoid()
          )
    
    forward(x):
        # x: (B, 3, 224, 224) in [0, 1]
        backbone_features = self.backbone(x)    # (B, 1792)
        spatial_features = self.feature_fc(backbone_features)  # (B, 256)
        prediction = self.classifier(spatial_features)  # (B, 1)
        return {
            "prediction": prediction,
            "spatial_features": spatial_features,
            "freq_features": None
        }
    """
    def __init__(self):
        super(EfficientNetDetector, self).__init__()
        
        # Load pretrained efficientnet_b4, without the classification head
        self.backbone = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
        
        # The output of EfficientNetB4 with num_classes=0 is (B, 1792)
        self.feature_fc = nn.Linear(1792, 256)
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        backbone_features = self.backbone(x) # (B, 1792)
        spatial_features = self.feature_fc(backbone_features) # (B, 256)
        
        prediction = self.classifier(spatial_features) # (B, 1)
        
        return {
            "prediction": prediction,
            "spatial_features": spatial_features,
            "freq_features": None
        }
