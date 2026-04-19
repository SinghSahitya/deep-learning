# Robust Deepfake Video Detection under Adversarial Conditions — Implementation Plan

## 1. Project Overview

Deepfake videos pose serious threats by enabling hyper-realistic misinformation. Current detectors achieve high accuracy on clean data but collapse under adversarial attacks (imperceptible pixel perturbations). This project builds an **adversarially robust deepfake detector** that integrates:

- **Adversarial Feature Similarity Learning** (from Khan et al. 2024)
- **Frequency-domain feature extraction** (motivated by hyperspectral deepfake detection literature)
- **Adversarial training** with PGD-based on-the-fly perturbations

The final system is a **dual-branch (spatial + frequency) classifier** trained with a combined loss that enforces robust internal representations.

---

## 2. Key Papers

| Paper | What We Use |
|-------|-------------|
| Khan et al. (2024) — Adversarial Feature Similarity Learning | AFS loss: enforces feature-space consistency between clean and adversarial inputs |
| Yan et al. (2024) — Latent Space Augmentation (CVPR 2024) | Motivation for forgery-agnostic feature learning and generalization |
| Exposing DeepFakes via Hyperspectral Domain Mapping (arXiv 2511.11732) | Motivation for frequency-domain branch in our architecture |

---

## 3. Dataset

### Primary: Celeb-DF v2

- **Source**: https://github.com/yuezunli/celeb-deepfakeforensics (request access via Google Form)
- **Contents**: ~600 real celebrity videos, ~5600 deepfake videos
- **Alternative sources**: Kaggle (search "Celeb-DF v2"), academic torrents
- **Preprocessing**: Extract frames from videos -> crop faces using MTCNN -> balance real/fake classes -> split 70/15/15 (train/val/test)
- **Storage estimate**: ~20-30 GB for extracted frames

### Optional (for cross-dataset generalization):

- **FaceForensics++**: https://github.com/ondyari/FaceForensics — 4 manipulation methods, good for generalization testing

---

## 4. Model Architecture

### Dual-Branch Multi-Domain Detector

```
Input Frame (224x224 RGB)
        |
        +---> [Spatial Branch: EfficientNet-B4 pretrained on ImageNet]
        |         |
        |         v
        |    Spatial Feature Vector (256-d)
        |
        +---> [Frequency Branch]
        |    Frame -> FFT 2D -> Log Magnitude Spectrum -> 3-layer CNN
        |         |
        |         v
        |    Frequency Feature Vector (128-d)
        |
        +---> Concatenate [Spatial(256) + Frequency(128)] = 384-d
                  |
                  v
             FC(384 -> 128) -> ReLU -> Dropout(0.3) -> FC(128 -> 1) -> Sigmoid
                  |
                  v
             Real / Fake prediction (0 = Real, 1 = Fake)
```

### Spatial Branch (EfficientNet-B4)

- Use `timm` library: `timm.create_model('efficientnet_b4', pretrained=True)`
- Remove the original classification head
- Add: `AdaptiveAvgPool2d(1)` -> `Linear(1792, 256)`
- This branch captures RGB-domain spatial artifacts

### Frequency Branch

- Input: 224x224 grayscale frame
- Apply `torch.fft.fft2` -> shift -> log(1 + magnitude)
- Pass the magnitude spectrum through a small CNN:
  - Conv2d(1, 32, 3, padding=1) -> BN -> ReLU -> MaxPool
  - Conv2d(32, 64, 3, padding=1) -> BN -> ReLU -> MaxPool
  - Conv2d(64, 128, 3, padding=1) -> BN -> ReLU -> AdaptiveAvgPool2d(1)
- Output: 128-d feature vector
- This branch captures frequency-domain inconsistencies introduced by deepfake generators

### Fusion Head

- Concatenate spatial (256-d) + frequency (128-d) = 384-d
- FC(384, 128) -> ReLU -> Dropout(0.3) -> FC(128, 1) -> Sigmoid

---

## 5. Loss Function

```
L_total = L_BCE + lambda_1 * L_AFS + lambda_2 * L_freq_consistency
```

### L_BCE — Binary Cross Entropy

Standard classification loss on clean + adversarial predictions.

### L_AFS — Adversarial Feature Similarity Loss (from Khan et al.)

```python
L_AFS = ||f_spatial(x_clean) - f_spatial(x_adv)||_2
```

Forces the spatial branch to produce similar feature representations for clean and adversarially perturbed versions of the same input. This is the core idea from the Khan et al. paper.

### L_freq_consistency — Frequency Feature Consistency Loss (OUR NOVELTY)

```python
L_freq_consistency = ||f_freq(x_clean) - f_freq(x_adv)||_2
```

Same idea applied to the frequency branch. Ensures the frequency features are also robust to adversarial perturbations. No prior paper applies feature similarity regularization to frequency-domain features for deepfake detection.

### Hyperparameters

- `lambda_1 = 0.5` (AFS weight, tune in [0.1, 0.5, 1.0])
- `lambda_2 = 0.3` (frequency consistency weight, tune in [0.1, 0.3, 0.5])
- Optimizer: Adam, LR = 1e-4, weight_decay = 1e-5
- Batch size: 32
- Epochs: 20-30

---

## 6. Adversarial Attacks

### FGSM (Fast Gradient Sign Method)

- Single-step attack
- `x_adv = x + epsilon * sign(grad_x(L))`
- Test with epsilon = {2/255, 4/255, 8/255}

### PGD (Projected Gradient Descent)

- Multi-step iterative attack (stronger than FGSM)
- Steps: 10, step size alpha = 1/255, epsilon = 4/255
- Random start within epsilon ball

### AutoAttack

- Ensemble of 4 attacks (APGD-CE, APGD-DLR, FAB, Square)
- Use the `autoattack` library
- Linf norm, epsilon = 4/255
- The strongest standardized benchmark for adversarial robustness

---

## 7. Training Procedure

### Phase 1: Baseline Training

- Train multi-domain model with only `L_BCE` on clean data
- No adversarial examples, no feature similarity loss
- This establishes baseline performance

### Phase 2: Adversarial Training

- For each training batch:
  1. Forward pass on clean images, get clean features and predictions
  2. Generate PGD adversarial examples (10 steps, epsilon=4/255)
  3. Forward pass on adversarial images, get adversarial features and predictions
  4. Compute `L_total = L_BCE(clean) + L_BCE(adv) + lambda_1 * L_AFS + lambda_2 * L_freq`
  5. Backprop and update
- Train for 20 epochs

### Phase 3: Fine-tuning

- Reduce LR to 1e-5
- Continue adversarial training for 5-10 more epochs

---

## 8. Experiments & Expected Results

### Main Results Table

| Model | Clean Acc | FGSM (eps=4/255) | PGD (eps=4/255) | AutoAttack | AUC |
|-------|-----------|-------------------|-----------------|------------|-----|
| Baseline CNN | ~95% | ~20-40% | ~10-30% | ~5-20% | ~0.95 |
| EfficientNet (no defense) | ~97% | ~25-45% | ~15-35% | ~10-25% | ~0.97 |
| + Adversarial Training | ~93% | ~70-80% | ~65-75% | ~55-65% | ~0.93 |
| + AFS Loss | ~94% | ~75-85% | ~70-80% | ~60-70% | ~0.94 |
| + Freq Branch (Full model) | ~94% | ~78-88% | ~72-82% | ~62-72% | ~0.94 |

### Ablation Study

| Configuration | Clean Acc | PGD Acc (eps=4/255) |
|---------------|-----------|---------------------|
| Full model (spatial + freq + AFS + adv training) | ~94% | ~75% |
| Remove frequency branch | ~93% | ~70% |
| Remove AFS loss | ~92% | ~65% |
| Remove adversarial training | ~96% | ~30% |
| lambda_1 = 0.1 | — | — |
| lambda_1 = 0.5 | — | — |
| lambda_1 = 1.0 | — | — |

### Figures to Generate

1. Confusion matrices (clean and adversarial conditions, for each model)
2. ROC curves comparing all model variants
3. t-SNE plots of feature embeddings (clean vs adversarial, baseline vs robust)
4. Adversarial perturbation visualizations (clean image, perturbation, adversarial image)
5. Training loss curves (all loss components over epochs)
6. Accuracy vs epsilon plot (robustness curve across different attack strengths)

---

## 9. Submission Folder Structure

```
Team2_RobustDeepfake/
+-- 01_admin/
|   +-- team_info.txt
|   +-- contribution_statement.pdf
+-- 02_report/
|   +-- final_report.pdf
|   +-- latex_source.zip
+-- 03_code/
|   +-- README.md
|   +-- requirements.txt
|   +-- src/
|   |   +-- data/
|   |   |   +-- dataset.py
|   |   |   +-- preprocessing.py
|   |   |   +-- augmentations.py
|   |   +-- models/
|   |   |   +-- baseline_cnn.py
|   |   |   +-- efficientnet_detector.py
|   |   |   +-- frequency_branch.py
|   |   |   +-- multi_domain_model.py
|   |   +-- attacks/
|   |   |   +-- fgsm.py
|   |   |   +-- pgd.py
|   |   |   +-- auto_attack.py
|   |   +-- losses/
|   |   |   +-- adversarial_feature_similarity.py
|   |   |   +-- combined_loss.py
|   |   +-- training/
|   |   |   +-- train_baseline.py
|   |   |   +-- train_adversarial.py
|   |   |   +-- evaluate.py
|   |   +-- utils/
|   |       +-- metrics.py
|   |       +-- visualization.py
|   |       +-- config.py
|   +-- scripts/
|   |   +-- preprocess_celebdf.py
|   |   +-- train.py
|   |   +-- eval.py
|   |   +-- demo.py
|   +-- configs/
|       +-- baseline.yaml
|       +-- adversarial_training.yaml
|       +-- multi_domain.yaml
+-- 04_data/
|   +-- data_description.md
|   +-- sample_inputs/
|   +-- dataset_links.txt
+-- 05_results/
|   +-- main_results.csv
|   +-- ablations.csv
|   +-- figures/
|   +-- logs/
+-- 06_demo/
|   +-- demo_instructions.md
|   +-- demo_inputs/
|   +-- backup_video.mp4
+-- 07_claims/
    +-- prior_work_basis.md
    +-- claimed_contribution.md
```

---

## 10. Code Structure Contracts

To enable parallel work, all team members must follow these interface contracts:

### Dataset Output Format

```python
# dataset.py returns items as:
{
    "image": torch.Tensor,    # shape (3, 224, 224), normalized to [0, 1]
    "label": torch.Tensor,    # 0 = real, 1 = fake
    "path": str               # original file path
}
```

### Model Forward Pass Contract

```python
# All models must implement:
class Model(nn.Module):
    def forward(self, x):
        # x: (B, 3, 224, 224)
        # returns: dict with keys:
        #   "prediction": (B, 1) sigmoid output
        #   "spatial_features": (B, 256) spatial branch features
        #   "freq_features": (B, 128) frequency branch features (or None for baseline)
        pass
```

### Attack Function Contract

```python
# All attack functions must follow:
def attack(model, images, labels, epsilon, **kwargs):
    # images: (B, 3, 224, 224) clean images
    # labels: (B,) ground truth
    # epsilon: float, perturbation budget
    # returns: (B, 3, 224, 224) adversarial images, clamped to [0, 1]
    pass
```

### Config YAML Format

```yaml
model:
  name: "multi_domain"        # or "baseline_cnn", "efficientnet"
  spatial_dim: 256
  freq_dim: 128

training:
  epochs: 20
  batch_size: 32
  lr: 0.0001
  weight_decay: 0.00001

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
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  image_size: 224
```

---

## 11. Tech Stack

```
Python 3.10+
PyTorch 2.0+
torchvision
timm                    # EfficientNet-B4 backbone
facenet-pytorch         # MTCNN for face detection/cropping
autoattack              # AutoAttack evaluation
gradio                  # Demo UI
numpy, pandas, scikit-learn
matplotlib, seaborn     # Plotting
pyyaml                  # Config files
tqdm                    # Progress bars
opencv-python           # Video frame extraction
```

---

## 12. Compute Requirements

- **GPU**: Minimum 1x NVIDIA GPU with 12GB+ VRAM
- **Free GPU options**: Google Colab (T4), Kaggle Notebooks (T4/P100, 30h/week)
- **Training time**: ~2-4 hours per model variant on T4
- **AutoAttack**: ~30 min for 1000 test samples — run on subset
- **Storage**: ~20-30 GB for extracted frames

---

## 13. Claimed Contributions (for the paper)

| Category | Description |
|----------|-------------|
| **Reproduced** | Adversarial Feature Similarity Learning (Khan et al. 2024) for deepfake detection |
| **Modified** | Extended AFS with a frequency-domain consistency loss |
| **Modified** | Added dual-branch (spatial + frequency) architecture instead of single-branch |
| **Contribution** | Combined multi-domain feature learning with adversarial robustness — showing frequency features provide additional resilience against adversarial attacks |

---

## 14. Team Division

- **Person A (Sahitya)**: Data pipeline, baseline models, demo, submission packaging
- **Person B (Nitin)**: Adversarial attacks, evaluation framework, results generation
- **Person C (Vishesh)**: Robust training, novel loss functions, frequency branch, ablations

Each person has a dedicated task file with full details.
