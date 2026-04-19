# Tasks — Vishesh (Person C)
## Role: Robust Training, Novel Loss Functions, Frequency Branch, Ablations
## Full Project Context
### What We Are Building

An **adversarially robust deepfake video detector**. The system is a dual-branch neural network that combines spatial features (EfficientNet-B4) and frequency features (FFT-based CNN) to classify video frames as real or fake. It is trained with a combined loss function that includes adversarial feature similarity regularization, making it robust against adversarial attacks (FGSM, PGD, AutoAttack).

### Architecture Overview

```
Input Frame (224x224 RGB)
        |
        +---> [Spatial Branch: EfficientNet-B4 pretrained]
        |         -> Spatial Feature Vector (256-d)
        |
        +---> [Frequency Branch: FFT -> Log Magnitude -> 3-layer CNN]
        |         -> Frequency Feature Vector (128-d)
        |
        +---> Concatenate [256 + 128] = 384-d
                  |
             FC(384->128) -> ReLU -> Dropout(0.3) -> FC(128->1) -> Sigmoid
                  |
             Real (0) / Fake (1)
```

### Loss Function

```
L_total = L_BCE + lambda_1 * L_AFS + lambda_2 * L_freq_consistency

L_AFS = ||spatial_features(clean) - spatial_features(adversarial)||_2
L_freq = ||freq_features(clean) - freq_features(adversarial)||_2
```

This combined loss is the core "novel" contribution of the project. The AFS loss comes from Khan et al. (2024). The frequency consistency loss applied to a dedicated frequency branch is our extension.

### Dataset

- **Celeb-DF v2**: ~600 real videos, ~5600 fake videos
- Preprocessed by Person A (Sahitya) into 224x224 face crops
- Split into train.csv, val.csv, test.csv with columns: path, label
- Labels: 0 = real, 1 = fake

### Code Structure

```
03_code/
+-- src/
|   +-- data/
|   |   +-- dataset.py              # Person A builds this
|   |   +-- preprocessing.py        # Person A builds this
|   |   +-- augmentations.py        # Person A builds this
|   +-- models/
|   |   +-- baseline_cnn.py         # Person A builds this
|   |   +-- efficientnet_detector.py # Person A builds this
|   |   +-- frequency_branch.py     # YOU BUILD THIS
|   |   +-- multi_domain_model.py   # YOU BUILD THIS
|   +-- attacks/
|   |   +-- fgsm.py                 # Person B builds this
|   |   +-- pgd.py                  # Person B builds this
|   |   +-- auto_attack.py          # Person B builds this
|   +-- losses/
|   |   +-- adversarial_feature_similarity.py  # YOU BUILD THIS
|   |   +-- combined_loss.py        # YOU BUILD THIS
|   +-- training/
|   |   +-- train_baseline.py       # Person A builds this
|   |   +-- train_adversarial.py    # YOU BUILD THIS
|   |   +-- evaluate.py            # Person B builds this
|   +-- utils/
|       +-- metrics.py             # Person B builds this
|       +-- visualization.py       # Person B builds this
|       +-- config.py              # Person A builds this
+-- scripts/
|   +-- preprocess_celebdf.py       # Person A builds this
|   +-- train.py                    # Person A builds this (you'll add adversarial mode)
|   +-- eval.py                     # Person B builds this
|   +-- demo.py                     # Person A builds this
+-- configs/
    +-- baseline.yaml               # Person A builds this
    +-- adversarial_training.yaml   # YOU BUILD THIS
    +-- multi_domain.yaml           # YOU BUILD THIS
```

### Interface Contracts (CRITICAL — follow these so everyone's code integrates)

**Dataset output format (Person A provides this):**
```python
{
    "image": torch.Tensor,    # shape (3, 224, 224), values in [0, 1]
    "label": torch.Tensor,    # scalar, 0 = real, 1 = fake
    "path": str
}
```

**Model forward pass contract (YOU MUST FOLLOW THIS for your models):**
```python
class Model(nn.Module):
    def forward(self, x):
        # x: (B, 3, 224, 224), values in [0, 1]
        # MUST return dict:
        #   "prediction": (B, 1) — sigmoid output
        #   "spatial_features": (B, 256) — spatial branch features before classifier
        #   "freq_features": (B, 128) — frequency branch features before classifier
        pass
```

**Attack function contract (Person B provides this, you import for training):**
```python
def pgd_attack(model, images, labels, epsilon, num_steps=10, alpha=None):
    # images: (B, 3, 224, 224) in [0, 1]
    # returns: (B, 3, 224, 224) adversarial images in [0, 1]
    pass
```

**Config YAML format:**
```yaml
model:
  name: "multi_domain"
  spatial_dim: 256
  freq_dim: 128

training:
  epochs: 20
  batch_size: 32
  lr: 0.0001
  weight_decay: 0.00001
  freeze_backbone_epochs: 5

loss:
  lambda_afs: 0.5
  lambda_freq: 0.3

adversarial:
  method: "pgd"
  epsilon: 0.01568           # 4/255
  pgd_steps: 10
  pgd_alpha: 0.00392         # 1/255

data:
  root: "./data/celeb_df_frames"
  train_csv: "./data/celeb_df_frames/train.csv"
  val_csv: "./data/celeb_df_frames/val.csv"
  test_csv: "./data/celeb_df_frames/test.csv"
  image_size: 224
  num_workers: 4
```

---

## Your Tasks

---

### TASK 1: Build Frequency Branch (`src/models/frequency_branch.py`)

**What**: A CNN that extracts features from the frequency domain (FFT magnitude spectrum) of face images. This is motivated by the observation that deepfake generators introduce frequency-domain inconsistencies that are harder for adversarial attacks to manipulate.

**File: `src/models/frequency_branch.py`**:

```python
import torch
import torch.nn as nn

class FrequencyBranch(nn.Module):
    """
    Extracts features from the frequency domain of input images.
    
    Pipeline:
    1. Convert RGB to grayscale: gray = 0.299*R + 0.587*G + 0.114*B
    2. Apply 2D FFT: spectrum = torch.fft.fft2(gray)
    3. Shift zero-frequency to center: spectrum = torch.fft.fftshift(spectrum)
    4. Compute log magnitude: mag = torch.log(1 + torch.abs(spectrum))
    5. Pass through CNN to extract features
    
    CNN architecture:
    - Conv2d(1, 32, kernel_size=3, padding=1) -> BatchNorm2d(32) -> ReLU -> MaxPool2d(2)
    - Conv2d(32, 64, kernel_size=3, padding=1) -> BatchNorm2d(64) -> ReLU -> MaxPool2d(2)
    - Conv2d(64, 128, kernel_size=3, padding=1) -> BatchNorm2d(128) -> ReLU -> AdaptiveAvgPool2d(1)
    - Flatten -> output shape: (B, 128)
    
    __init__(self, output_dim=128):
        # Build the CNN layers
        # output_dim: dimension of the output feature vector
    
    forward(self, x):
        # x: (B, 3, 224, 224) RGB images in [0, 1]
        # Returns: (B, 128) frequency features
        
        # Step 1: RGB to grayscale
        gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        # gray shape: (B, 1, 224, 224)
        
        # Step 2-4: FFT -> shift -> log magnitude
        spectrum = torch.fft.fft2(gray)
        spectrum = torch.fft.fftshift(spectrum)
        magnitude = torch.log(1 + torch.abs(spectrum))
        # magnitude shape: (B, 1, 224, 224)
        
        # Step 5: CNN
        features = self.cnn(magnitude)  # (B, 128)
        return features
    """
```

**Why frequency features matter**:
- Deepfake generators (GANs, diffusion models) leave artifacts in the frequency domain
- These artifacts are harder for adversarial perturbations to modify (perturbations are typically small in spatial domain and spread out in frequency domain)
- The FFT operation itself is differentiable in PyTorch, so gradients flow through for end-to-end training

**Important**: Make sure the FFT operations work on GPU tensors. `torch.fft.fft2` supports CUDA tensors natively.

---

### TASK 2: Build Multi-Domain Model (`src/models/multi_domain_model.py`)

**What**: The full dual-branch model that combines EfficientNet spatial features with frequency features.

**File: `src/models/multi_domain_model.py`**:

```python
import torch
import torch.nn as nn
import timm
from .frequency_branch import FrequencyBranch

class MultiDomainDetector(nn.Module):
    """
    Dual-branch deepfake detector combining spatial and frequency features.
    
    __init__(self, spatial_dim=256, freq_dim=128, pretrained=True):
        # Spatial branch: EfficientNet-B4
        self.backbone = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=0)
        # backbone outputs (B, 1792)
        self.spatial_fc = nn.Linear(1792, spatial_dim)  # (B, 256)
        
        # Frequency branch
        self.freq_branch = FrequencyBranch(output_dim=freq_dim)  # (B, 128)
        
        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(spatial_dim + freq_dim, 128),  # 384 -> 128
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    forward(self, x):
        # x: (B, 3, 224, 224) in [0, 1]
        
        # Spatial branch
        backbone_out = self.backbone(x)           # (B, 1792)
        spatial_features = self.spatial_fc(backbone_out)  # (B, 256)
        
        # Frequency branch
        freq_features = self.freq_branch(x)       # (B, 128)
        
        # Fusion
        combined = torch.cat([spatial_features, freq_features], dim=1)  # (B, 384)
        prediction = self.classifier(combined)     # (B, 1)
        
        return {
            "prediction": prediction,
            "spatial_features": spatial_features,
            "freq_features": freq_features
        }
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
    """
```

**Important design decisions**:
- The frequency branch takes the SAME input as the spatial branch (the original RGB image), not a pre-computed FFT. The FFT is computed inside the branch so gradients can flow through for adversarial training.
- Both branches produce features independently, then concatenate. No cross-attention or complex fusion — keep it simple.
- The `spatial_features` and `freq_features` outputs are used by the loss function (AFS and freq consistency loss) during training.

---

### TASK 3: Implement Adversarial Feature Similarity Loss (`src/losses/adversarial_feature_similarity.py`)

**What**: The AFS loss from Khan et al. (2024). This forces the model's internal features to be similar for clean and adversarially perturbed versions of the same image.

**File: `src/losses/adversarial_feature_similarity.py`**:

```python
import torch
import torch.nn as nn

class AdversarialFeatureSimilarityLoss(nn.Module):
    """
    AFS Loss: Minimizes the L2 distance between features of clean and adversarial inputs.
    
    From Khan et al. (2024): "Adversarially Robust Deepfake Detection via 
    Adversarial Feature Similarity Learning"
    
    L_AFS = (1/B) * sum_i ||f(x_i) - f(x_i_adv)||_2
    
    where:
    - f(x_i) = feature vector for clean input i
    - f(x_i_adv) = feature vector for adversarial input i
    - B = batch size
    
    This loss encourages the model to learn representations that are 
    INVARIANT to adversarial perturbations.
    
    forward(self, clean_features, adv_features):
        # clean_features: (B, D) features from clean images
        # adv_features: (B, D) features from adversarial images
        # Returns: scalar loss value
        
        # Compute per-sample L2 distance
        distances = torch.norm(clean_features - adv_features, p=2, dim=1)  # (B,)
        # Return mean distance
        return distances.mean()
    """
```

**Why this works**: Without this loss, the model might achieve high clean accuracy but have wildly different internal representations for clean vs adversarial inputs. By penalizing feature-space distance, the model is forced to be "smooth" — small input perturbations cause small feature changes, making the final classification more stable.

---

### TASK 4: Implement Combined Loss (`src/losses/combined_loss.py`)

**What**: The full training objective combining classification loss, AFS loss, and frequency consistency loss.

**File: `src/losses/combined_loss.py`**:

```python
import torch
import torch.nn as nn
from .adversarial_feature_similarity import AdversarialFeatureSimilarityLoss

class CombinedRobustLoss(nn.Module):
    """
    Combined loss for adversarially robust deepfake detection.
    
    L_total = L_BCE_clean + L_BCE_adv + lambda_afs * L_AFS_spatial + lambda_freq * L_AFS_freq
    
    __init__(self, lambda_afs=0.5, lambda_freq=0.3):
        self.bce = nn.BCELoss()
        self.afs = AdversarialFeatureSimilarityLoss()
        self.lambda_afs = lambda_afs
        self.lambda_freq = lambda_freq
    
    forward(self, clean_output, adv_output, labels):
        # clean_output: dict from model.forward(clean_images)
        #   {"prediction": (B,1), "spatial_features": (B,256), "freq_features": (B,128)}
        # adv_output: dict from model.forward(adv_images)
        #   {"prediction": (B,1), "spatial_features": (B,256), "freq_features": (B,128)}
        # labels: (B,) ground truth
        
        labels_float = labels.float().unsqueeze(1)  # (B, 1)
        
        # Classification loss on both clean and adversarial
        bce_clean = self.bce(clean_output["prediction"], labels_float)
        bce_adv = self.bce(adv_output["prediction"], labels_float)
        
        # Spatial AFS loss (from Khan et al.)
        afs_spatial = self.afs(
            clean_output["spatial_features"],
            adv_output["spatial_features"]
        )
        
        # Frequency consistency loss (OUR CONTRIBUTION)
        afs_freq = self.afs(
            clean_output["freq_features"],
            adv_output["freq_features"]
        )
        
        total = bce_clean + bce_adv + self.lambda_afs * afs_spatial + self.lambda_freq * afs_freq
        
        # Return total and components for logging
        return {
            "total": total,
            "bce_clean": bce_clean.item(),
            "bce_adv": bce_adv.item(),
            "afs_spatial": afs_spatial.item(),
            "afs_freq": afs_freq.item()
        }
    """
```

**Note on the novelty**: The `afs_freq` term (frequency feature consistency loss) is what distinguishes our approach from Khan et al. They only apply AFS to the spatial features. We extend it to frequency features AND add a dedicated frequency extraction branch. This is a defensible and honest contribution.

---

### TASK 5: Build Adversarial Training Loop (`src/training/train_adversarial.py`)

**What**: The training loop that generates adversarial examples on-the-fly and trains with the combined loss.

**File: `src/training/train_adversarial.py`**:

```python
import torch
from src.attacks.pgd import pgd_attack
from src.losses.combined_loss import CombinedRobustLoss

def train_adversarial(model, train_loader, val_loader, config, device):
    """
    Adversarial training loop.
    
    For each epoch:
        For each batch:
            1. Get clean images and labels from batch
            2. Forward pass on clean images:
               clean_output = model(clean_images)
            3. Generate adversarial examples using PGD:
               adv_images = pgd_attack(model, clean_images, labels, 
                                       epsilon=config.adversarial.epsilon,
                                       num_steps=config.adversarial.pgd_steps,
                                       alpha=config.adversarial.pgd_alpha)
            4. Forward pass on adversarial images:
               adv_output = model(adv_images)
            5. Compute combined loss:
               loss_dict = criterion(clean_output, adv_output, labels)
            6. Backward pass on loss_dict["total"]
            7. Optimizer step
            8. Log all loss components
        
        Validation:
            - Compute clean accuracy on val set
            - Compute PGD accuracy on val set (with fewer steps for speed, e.g., 5 steps)
            - Save best model by robust val accuracy (PGD accuracy)
    
    Args:
        model: MultiDomainDetector (or EfficientNetDetector for ablation)
        train_loader: DataLoader
        val_loader: DataLoader
        config: dict from YAML
        device: 'cuda' or 'cpu'
    
    Returns:
        history: dict with per-epoch logs of all loss components and val metrics
    """
```

**Critical implementation details**:

1. **Model must be in train mode during PGD generation**: This is counterintuitive. Normally attacks use eval mode. But during adversarial training, you need BatchNorm in train mode for consistent statistics. Set `model.train()` before the entire training step.

2. **Detach adversarial images before the forward pass**: After generating `adv_images` with PGD, call `adv_images = adv_images.detach()` before the second forward pass. The PGD generation already computes gradients through the model — you don't want to backprop through the attack generation.

3. **Backbone freezing**: For the first `freeze_backbone_epochs` epochs, call `model.freeze_backbone()`. After that, call `model.unfreeze_backbone()`. This prevents the pretrained EfficientNet from being destroyed early in training.

4. **Learning rate schedule**: Use CosineAnnealingLR or ReduceLROnPlateau. Start at 1e-4, minimum 1e-6.

5. **Gradient clipping**: Use `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` to prevent gradient explosion from the adversarial training.

6. **Save checkpoints**: Save best model by PGD val accuracy (not clean val accuracy), because we care about robustness.

---

### TASK 6: Create Config Files

**File: `configs/adversarial_training.yaml`**:
```yaml
model:
  name: "efficientnet"
  spatial_dim: 256

training:
  epochs: 25
  batch_size: 32
  lr: 0.0001
  weight_decay: 0.00001
  freeze_backbone_epochs: 5
  grad_clip: 1.0
  scheduler: "cosine"

loss:
  lambda_afs: 0.5
  lambda_freq: 0.0    # No freq branch in this config (ablation)

adversarial:
  method: "pgd"
  epsilon: 0.01568     # 4/255
  pgd_steps: 10
  pgd_alpha: 0.00392   # 1/255
  val_pgd_steps: 5     # Fewer steps during validation for speed

data:
  root: "./data/celeb_df_frames"
  train_csv: "./data/celeb_df_frames/train.csv"
  val_csv: "./data/celeb_df_frames/val.csv"
  test_csv: "./data/celeb_df_frames/test.csv"
  image_size: 224
  num_workers: 4
```

**File: `configs/multi_domain.yaml`**:
```yaml
model:
  name: "multi_domain"
  spatial_dim: 256
  freq_dim: 128

training:
  epochs: 25
  batch_size: 32
  lr: 0.0001
  weight_decay: 0.00001
  freeze_backbone_epochs: 5
  grad_clip: 1.0
  scheduler: "cosine"

loss:
  lambda_afs: 0.5
  lambda_freq: 0.3

adversarial:
  method: "pgd"
  epsilon: 0.01568
  pgd_steps: 10
  pgd_alpha: 0.00392
  val_pgd_steps: 5

data:
  root: "./data/celeb_df_frames"
  train_csv: "./data/celeb_df_frames/train.csv"
  val_csv: "./data/celeb_df_frames/val.csv"
  test_csv: "./data/celeb_df_frames/test.csv"
  image_size: 224
  num_workers: 4
```

---

### TASK 7: Train All Robust Model Variants

Once Person A delivers the preprocessed dataset and Person B delivers the PGD attack code, train the following model variants. Each produces a checkpoint that Person B will evaluate.

**Variant 1: EfficientNet + Adversarial Training only (no AFS, no freq)**
```bash
python scripts/train.py --config configs/adversarial_training.yaml
# Uses: BCELoss on clean + adversarial, lambda_afs=0, lambda_freq=0
# Save as: checkpoints/efficientnet_advtrain.pth
```
For this variant, modify the combined loss to only use BCE terms (set both lambdas to 0 in config).

**Variant 2: EfficientNet + Adversarial Training + AFS Loss (no freq)**
```bash
python scripts/train.py --config configs/adversarial_training.yaml
# With lambda_afs=0.5, lambda_freq=0
# Save as: checkpoints/efficientnet_afs.pth
```

**Variant 3: Full Multi-Domain Model (spatial + freq + AFS + freq consistency)**
```bash
python scripts/train.py --config configs/multi_domain.yaml
# With lambda_afs=0.5, lambda_freq=0.3
# Save as: checkpoints/multi_domain_full.pth
```

**Give all three checkpoints to Person B for evaluation.**

---

### TASK 8: Run Ablation Experiments

**What**: Systematically remove or vary each component to show its individual contribution.

**Ablations to run:**

| Experiment | Model | lambda_afs | lambda_freq | Notes |
|------------|-------|------------|-------------|-------|
| Full model | multi_domain | 0.5 | 0.3 | Best expected result |
| No freq branch | efficientnet | 0.5 | 0.0 | Shows freq branch contribution |
| No AFS loss | multi_domain | 0.0 | 0.3 | Shows AFS contribution |
| No adv training | multi_domain | 0.0 | 0.0 | Trained on clean data only with freq branch |
| lambda_afs = 0.1 | multi_domain | 0.1 | 0.3 | Hyperparameter sensitivity |
| lambda_afs = 1.0 | multi_domain | 1.0 | 0.3 | Hyperparameter sensitivity |
| lambda_freq = 0.1 | multi_domain | 0.5 | 0.1 | Hyperparameter sensitivity |
| lambda_freq = 0.5 | multi_domain | 0.5 | 0.5 | Hyperparameter sensitivity |

For each ablation:
1. Train the model variant
2. Evaluate clean accuracy and PGD accuracy (eps=4/255)
3. Record results

**Save results as `05_results/ablations.csv`** with columns: experiment, model, lambda_afs, lambda_freq, clean_acc, pgd_acc

---

### TASK 9: Prepare Claims Documents

**File: `07_claims/prior_work_basis.md`**:

```markdown
# Prior Work Basis

## Paper 1: Khan et al. (2024) - Adversarial Feature Similarity Learning
- Introduced the concept of Adversarial Feature Similarity (AFS) loss
- We reproduced their loss function and applied it to our EfficientNet-B4 backbone
- Our implementation follows their formulation: L_AFS = mean(||f(x) - f(x_adv)||_2)

## Paper 2: Yan et al. (2024) - Latent Space Augmentation (CVPR 2024)
- Motivated our approach to learning forgery-agnostic features
- We did not directly implement their augmentation method but their insight
  about generalization influenced our multi-domain feature design

## Paper 3: Exposing DeepFakes via Hyperspectral Domain Mapping
- Motivated our frequency-domain branch
- Their observation that frequency-domain inconsistencies are robust to 
  common perturbations led us to incorporate FFT-based feature extraction
```

**File: `07_claims/claimed_contribution.md`**:

```markdown
# Claimed Contributions

## What We Reproduced
- Adversarial Feature Similarity Loss from Khan et al. (2024)
- Standard adversarial training with PGD for deepfake detection
- Evaluation under FGSM, PGD, and AutoAttack

## What We Modified
- Extended AFS loss to frequency-domain features (original paper only uses spatial)
- Added a dedicated frequency feature extraction branch (FFT + CNN)
- Combined spatial and frequency branches in a dual-branch architecture

## What Did Not Work
[Fill in honestly after experiments — e.g., certain lambda values, 
higher epsilon training, specific architectures tried and abandoned]

## What We Believe Is Our Contribution
- A dual-branch (spatial + frequency) architecture for adversarially robust 
  deepfake detection
- Frequency Feature Consistency Loss: applying adversarial feature similarity 
  regularization to frequency-domain features
- Empirical evidence that frequency features provide additional resilience 
  against adversarial attacks beyond what spatial-only AFS provides
```

---

## Dependencies You Need to Install

```bash
pip install torch torchvision timm pyyaml tqdm pandas numpy
```

---

## What You Need from Teammates

- **From Person A (Sahitya)**: Preprocessed dataset (face images + CSV splits), EfficientNet model code (you import and extend it), DataLoader code
- **From Person B (Nitin)**: PGD attack function (you import it for adversarial training)

## What You Deliver

- Frequency branch code (frequency_branch.py)
- Multi-domain model code (multi_domain_model.py)
- All loss functions (adversarial_feature_similarity.py, combined_loss.py)
- Adversarial training loop (train_adversarial.py)
- Config files (adversarial_training.yaml, multi_domain.yaml)
- All trained robust model checkpoints → give to Person B
- Ablation results → give to Person B for inclusion in results
- Claims documents (07_claims/)
