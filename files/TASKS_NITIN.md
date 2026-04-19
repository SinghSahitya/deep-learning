# Tasks — Nitin (Person B)

## Role: Adversarial Attacks, Evaluation Framework, Results & Visualization

---

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
L_total = L_BCE + 0.5 * L_AFS + 0.3 * L_freq_consistency

L_AFS = ||spatial_features(clean) - spatial_features(adversarial)||_2
L_freq = ||freq_features(clean) - freq_features(adversarial)||_2
```

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
|   |   +-- frequency_branch.py     # Person C builds this
|   |   +-- multi_domain_model.py   # Person C builds this
|   +-- attacks/
|   |   +-- fgsm.py                 # YOU BUILD THIS
|   |   +-- pgd.py                  # YOU BUILD THIS
|   |   +-- auto_attack.py          # YOU BUILD THIS
|   +-- losses/                     # Person C builds these
|   +-- training/
|   |   +-- train_baseline.py       # Person A builds this
|   |   +-- train_adversarial.py    # Person C builds this
|   |   +-- evaluate.py            # YOU BUILD THIS
|   +-- utils/
|       +-- metrics.py             # YOU BUILD THIS
|       +-- visualization.py       # YOU BUILD THIS
|       +-- config.py              # Person A builds this
+-- scripts/
|   +-- preprocess_celebdf.py       # Person A builds this
|   +-- train.py                    # Person A builds this
|   +-- eval.py                     # YOU BUILD THIS
|   +-- demo.py                     # Person A builds this
+-- configs/
    +-- baseline.yaml               # Person A builds this
    +-- adversarial_training.yaml   # Person C builds this
    +-- multi_domain.yaml           # Person C builds this
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

**Model forward pass contract (all models follow this):**
```python
class Model(nn.Module):
    def forward(self, x):
        # x: (B, 3, 224, 224), values in [0, 1]
        # Returns dict:
        #   "prediction": (B, 1) — sigmoid output
        #   "spatial_features": (B, 256)
        #   "freq_features": (B, 128) or None for models without freq branch
        pass
```

**YOUR attack function contract (Person C's training code will import these):**
```python
def attack(model, images, labels, epsilon, **kwargs):
    """
    Args:
        model: nn.Module following the forward pass contract above
        images: (B, 3, 224, 224), clean images in [0, 1]
        labels: (B,) ground truth labels
        epsilon: float, perturbation budget (e.g., 4/255 = 0.01568)
        **kwargs: attack-specific params (e.g., num_steps, alpha for PGD)
    
    Returns:
        adversarial_images: (B, 3, 224, 224), clamped to [0, 1]
    """
```

---

## Your Tasks

---

### TASK 1: Implement FGSM Attack (`src/attacks/fgsm.py`)

**What**: Fast Gradient Sign Method — a single-step adversarial attack.

**Algorithm**:
```
1. Set model to eval mode
2. Clone images, set requires_grad=True
3. Forward pass: output = model(images)["prediction"]
4. Compute loss: loss = BCELoss(output.squeeze(), labels.float())
5. Backward: loss.backward()
6. Compute perturbation: perturbation = epsilon * sign(images.grad)
7. adversarial_images = images + perturbation
8. Clamp to [0, 1]
9. Return adversarial_images (detached)
```

**File: `src/attacks/fgsm.py`**:

```python
import torch
import torch.nn as nn

def fgsm_attack(model, images, labels, epsilon):
    """
    FGSM attack implementation.
    
    Args:
        model: deepfake detector model
        images: (B, 3, 224, 224) clean images in [0, 1]
        labels: (B,) ground truth (0=real, 1=fake)
        epsilon: perturbation budget (e.g., 4/255)
    
    Returns:
        (B, 3, 224, 224) adversarial images clamped to [0, 1]
    """
    # IMPORTANT: The model uses Sigmoid output, so use BCELoss (not CrossEntropyLoss)
    # IMPORTANT: Don't modify the model's training state — save and restore it
    # IMPORTANT: Detach the result and don't retain the computation graph
    pass
```

**Test with**: epsilon values of 2/255, 4/255, 8/255. At 8/255, even good models should show significant accuracy drops.

---

### TASK 2: Implement PGD Attack (`src/attacks/pgd.py`)

**What**: Projected Gradient Descent — a multi-step iterative attack, much stronger than FGSM.

**Algorithm**:
```
1. Set model to eval mode
2. Initialize: adv_images = images + uniform_random(-epsilon, epsilon), clamp to [0, 1]
3. For step in range(num_steps):
    a. adv_images.requires_grad = True
    b. output = model(adv_images)["prediction"]
    c. loss = BCELoss(output.squeeze(), labels.float())
    d. loss.backward()
    e. adv_images = adv_images + alpha * sign(adv_images.grad)
    f. # Project back: perturbation = clamp(adv_images - images, -epsilon, epsilon)
    g. adv_images = clamp(images + perturbation, 0, 1)
    h. adv_images = adv_images.detach()
4. Return adv_images
```

**File: `src/attacks/pgd.py`**:

```python
import torch
import torch.nn as nn

def pgd_attack(model, images, labels, epsilon, num_steps=10, alpha=None):
    """
    PGD attack implementation.
    
    Args:
        model: deepfake detector model
        images: (B, 3, 224, 224) clean images in [0, 1]
        labels: (B,) ground truth
        epsilon: perturbation budget (e.g., 4/255)
        num_steps: number of PGD iterations (default: 10)
        alpha: step size per iteration (default: epsilon/4 if None)
    
    Returns:
        (B, 3, 224, 224) adversarial images clamped to [0, 1]
    """
    # Default alpha
    if alpha is None:
        alpha = epsilon / 4
    
    # IMPORTANT: Random start within epsilon ball (this is what distinguishes PGD from I-FGSM)
    # IMPORTANT: Project back onto epsilon-ball AND [0,1] range at every step
    # IMPORTANT: Detach at each step to avoid graph buildup
    pass
```

**Default params**: epsilon=4/255, num_steps=10, alpha=1/255. These are standard in the adversarial robustness literature.

---

### TASK 3: Implement AutoAttack Wrapper (`src/attacks/auto_attack.py`)

**What**: Wrapper around the `autoattack` library — the gold standard for evaluating adversarial robustness.

**File: `src/attacks/auto_attack.py`**:

```python
from autoattack import AutoAttack

def auto_attack_eval(model, images, labels, epsilon, batch_size=32):
    """
    Run AutoAttack evaluation.
    
    Args:
        model: deepfake detector model
        images: (N, 3, 224, 224) ALL test images (not just one batch)
        labels: (N,) ground truth
        epsilon: perturbation budget
        batch_size: internal batch size for AutoAttack
    
    Returns:
        dict with:
            "adversarial_images": (N, 3, 224, 224)
            "robust_accuracy": float
            "clean_accuracy": float
    """
    # AutoAttack expects:
    # - model that takes (B, 3, H, W) and returns (B, num_classes) LOGITS (not sigmoid!)
    # - So you need a wrapper class:
    
    class ModelWrapper(nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.model = original_model
        
        def forward(self, x):
            out = self.model(x)["prediction"]  # (B, 1) sigmoid
            # Convert to 2-class logits for AutoAttack:
            # fake_logit = logit(sigmoid_output), real_logit = logit(1 - sigmoid_output)
            fake_prob = out.squeeze(1)
            real_prob = 1 - fake_prob
            # Use log to convert to logit-like scale
            logits = torch.stack([
                torch.log(real_prob + 1e-8),
                torch.log(fake_prob + 1e-8)
            ], dim=1)  # (B, 2)
            return logits
    
    # Then:
    # wrapped = ModelWrapper(model)
    # adversary = AutoAttack(wrapped, norm='Linf', eps=epsilon, version='standard')
    # adversary.run_standard_evaluation(images, labels, bs=batch_size)
    pass
```

**Important notes**:
- AutoAttack is SLOW. On 1000 images it can take 30+ minutes on a T4 GPU.
- Run it on a subset of the test set (e.g., 500-1000 images) to save time.
- Install via `pip install autoattack`.
- The model wrapper converting sigmoid to logits is critical — AutoAttack expects multi-class logit outputs.

---

### TASK 4: Build Evaluation Framework (`src/training/evaluate.py`)

**What**: Comprehensive evaluation function that tests any model under clean and adversarial conditions.

**File: `src/training/evaluate.py`**:

```python
def evaluate_model(model, test_loader, device, attacks=None):
    """
    Evaluate model on clean data and under adversarial attacks.
    
    Args:
        model: detector model
        test_loader: DataLoader for test set
        device: 'cuda' or 'cpu'
        attacks: dict of attack_name -> attack_function
                 e.g., {"fgsm_2": lambda m,x,y: fgsm_attack(m,x,y,2/255),
                        "fgsm_4": lambda m,x,y: fgsm_attack(m,x,y,4/255),
                        "pgd_4": lambda m,x,y: pgd_attack(m,x,y,4/255)}
    
    Returns:
        dict with results:
        {
            "clean": {"accuracy": float, "auc": float, "predictions": list, "labels": list},
            "fgsm_2": {"accuracy": float, "auc": float, ...},
            "fgsm_4": {"accuracy": float, "auc": float, ...},
            ...
        }
    """
    # For each condition (clean + each attack):
    # 1. Get predictions for all test samples
    # 2. Compute accuracy (threshold 0.5), AUC, confusion matrix
    # 3. Store predictions and labels for later visualization
    pass


def run_full_evaluation(model, test_loader, device):
    """
    Convenience function that runs all standard attacks.
    Defines the attack dict with FGSM (eps=2,4,8/255), PGD (eps=4/255), 
    and calls evaluate_model.
    Returns results dict.
    """
    pass
```

---

### TASK 5: Build Metrics Utilities (`src/utils/metrics.py`)

**What**: Functions to compute all metrics needed for results tables.

**File: `src/utils/metrics.py`**:

```python
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve

def compute_metrics(predictions, labels, threshold=0.5):
    """
    Compute all classification metrics.
    
    Args:
        predictions: list/array of sigmoid outputs (floats)
        labels: list/array of ground truth (0 or 1)
        threshold: classification threshold
    
    Returns:
        dict: {
            "accuracy": float,
            "auc": float,
            "confusion_matrix": 2x2 array,
            "precision": float,
            "recall": float,
            "f1": float,
            "fpr": array (for ROC curve),
            "tpr": array (for ROC curve)
        }
    """
    pass

def format_results_table(results_dict):
    """
    Take results from evaluate_model() and format as a pandas DataFrame.
    Rows: model variants
    Columns: Clean Acc, FGSM (eps=2), FGSM (eps=4), FGSM (eps=8), PGD (eps=4), AUC
    
    Returns: pandas DataFrame
    """
    pass

def save_results_csv(results_df, path):
    """Save results DataFrame to CSV."""
    results_df.to_csv(path, index=True)
```

---

### TASK 6: Build Visualization Module (`src/utils/visualization.py`)

**What**: Generate all figures needed for the paper and presentation.

**File: `src/utils/visualization.py`**:

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE

def plot_confusion_matrix(cm, title, save_path):
    """
    Plot a 2x2 confusion matrix heatmap.
    - Labels: ["Real", "Fake"] on both axes
    - Show counts in each cell
    - Title: e.g., "Baseline CNN - Clean Data" or "Full Model - PGD Attack"
    - Save to save_path as PNG (300 DPI)
    """
    pass

def plot_roc_curves(results_dict, save_path):
    """
    Plot ROC curves for multiple models/conditions on the same figure.
    
    Args:
        results_dict: {model_name: {"fpr": array, "tpr": array, "auc": float}, ...}
    
    - Each model gets a different color line
    - Include AUC in legend: "Model Name (AUC = 0.95)"
    - Include diagonal reference line
    - Save as PNG
    """
    pass

def plot_accuracy_vs_epsilon(model_results, save_path):
    """
    Plot accuracy vs epsilon for a single model across FGSM attacks.
    
    Args:
        model_results: {epsilon_value: accuracy, ...} 
                       e.g., {0: 0.95, 2/255: 0.80, 4/255: 0.65, 8/255: 0.40}
    
    - X-axis: epsilon (0, 2/255, 4/255, 8/255)
    - Y-axis: accuracy
    - Plot multiple models as different lines for comparison
    """
    pass

def plot_tsne(features_clean, features_adv, labels, save_path):
    """
    t-SNE visualization of feature embeddings.
    
    Args:
        features_clean: (N, D) clean features from model
        features_adv: (N, D) adversarial features from model
        labels: (N,) ground truth
    
    - Combine clean + adversarial features
    - Run t-SNE to 2D
    - Plot with 4 colors: real-clean, fake-clean, real-adv, fake-adv
    - This shows how adversarial training keeps features close together
    """
    pass

def plot_adversarial_examples(clean_images, adv_images, perturbations, predictions_clean, predictions_adv, save_path):
    """
    Show a grid of adversarial example visualizations.
    
    For each example (show 4-6 examples):
    - Row: [Clean Image | Perturbation (amplified 10x) | Adversarial Image]
    - Caption: "Clean: Real(0.95) | Adv: Fake(0.62)"
    
    This is a great figure for the paper — shows the attack is imperceptible
    but fools the non-robust model.
    """
    pass

def plot_training_curves(history, save_path):
    """
    Plot training loss curves.
    
    Args:
        history: dict with keys like "train_loss", "val_loss", "val_accuracy"
                 each is a list of values per epoch
    
    - Subplot 1: Train loss + Val loss over epochs
    - Subplot 2: Val accuracy over epochs
    """
    pass
```

---

### TASK 7: Build Evaluation Script (`scripts/eval.py`)

**What**: Command-line entry point for running full evaluations.

**File: `scripts/eval.py`**:

```python
"""
Usage: python scripts/eval.py --checkpoint path/to/model.pth --config configs/baseline.yaml --output_dir 05_results/

This script:
1. Loads a trained model checkpoint
2. Loads the test dataset
3. Runs clean evaluation
4. Runs FGSM attacks at eps = 2/255, 4/255, 8/255
5. Runs PGD attack at eps = 4/255
6. (Optional) Runs AutoAttack at eps = 4/255 on a subset
7. Computes all metrics
8. Saves:
   - results to main_results.csv (or appends a row if file exists)
   - confusion matrices for each condition to figures/
   - ROC curves to figures/
   - adversarial example visualizations to figures/
   
Command-line args:
  --checkpoint: path to .pth file
  --config: path to YAML config
  --model_name: name for this model in results table (e.g., "Baseline CNN", "Full Model")
  --output_dir: where to save results (default: 05_results/)
  --run_autoattack: flag to enable AutoAttack (slow, off by default)
  --autoattack_samples: number of samples for AutoAttack (default: 500)
  --device: cuda or cpu
"""
```

---

### TASK 8: Generate All Final Results

**What**: Run evaluations on all model checkpoints and produce the final results tables and figures.

Once all models are trained (by Person A for baselines, Person C for robust models), run this sequence:

```bash
# Evaluate each model variant
python scripts/eval.py --checkpoint checkpoints/baseline_cnn.pth --config configs/baseline.yaml --model_name "Baseline CNN"
python scripts/eval.py --checkpoint checkpoints/efficientnet.pth --config configs/baseline.yaml --model_name "EfficientNet (no defense)"
python scripts/eval.py --checkpoint checkpoints/efficientnet_advtrain.pth --config configs/adversarial_training.yaml --model_name "+ Adversarial Training"
python scripts/eval.py --checkpoint checkpoints/efficientnet_afs.pth --config configs/adversarial_training.yaml --model_name "+ AFS Loss"
python scripts/eval.py --checkpoint checkpoints/multi_domain_full.pth --config configs/multi_domain.yaml --model_name "Full Model (Ours)"

# Run AutoAttack on key models (slow)
python scripts/eval.py --checkpoint checkpoints/efficientnet.pth --config configs/baseline.yaml --model_name "EfficientNet" --run_autoattack --autoattack_samples 500
python scripts/eval.py --checkpoint checkpoints/multi_domain_full.pth --config configs/multi_domain.yaml --model_name "Full Model" --run_autoattack --autoattack_samples 500
```

**Files to produce in `05_results/`**:

```
05_results/
+-- main_results.csv              # All models, all attacks, all metrics
+-- ablations.csv                 # Ablation study results (from Person C's experiments)
+-- figures/
|   +-- confusion_matrix_baseline_clean.png
|   +-- confusion_matrix_baseline_pgd.png
|   +-- confusion_matrix_fullmodel_clean.png
|   +-- confusion_matrix_fullmodel_pgd.png
|   +-- roc_curves_all_models.png
|   +-- accuracy_vs_epsilon.png
|   +-- tsne_baseline.png
|   +-- tsne_fullmodel.png
|   +-- adversarial_examples.png
|   +-- training_curves.png
+-- logs/
    +-- eval_baseline_cnn.json
    +-- eval_efficientnet.json
    +-- eval_full_model.json
```

---

### TASK 9: Generate t-SNE Feature Visualizations

**What**: Extract features from models and create t-SNE plots showing how adversarial training keeps clean and adversarial features close.

```python
"""
For both baseline and full model:
1. Load model
2. Run test set through model, collect spatial_features for all samples
3. Run PGD attack on test set, collect spatial_features for adversarial samples
4. Run t-SNE on combined [clean_features; adv_features]
5. Plot with 4 colors: real-clean, fake-clean, real-adv, fake-adv

The baseline model's t-SNE should show adversarial features scattered/shifted.
The full model's t-SNE should show adversarial features overlapping with clean features.
This is a very compelling visualization for the paper.
"""
```

---

## Dependencies You Need to Install

```bash
pip install torch torchvision autoattack scikit-learn matplotlib seaborn pandas numpy tqdm
```

---

## What You Need from Teammates

- **From Person A (Sahitya)**: Preprocessed dataset (face images + CSV splits) + trained baseline model checkpoints (baseline_cnn.pth, efficientnet.pth) + working DataLoader code
- **From Person C (Vishesh)**: All robust model checkpoints (advtrain, afs, full model) + ablation results

## What You Deliver

- All attack implementations (fgsm.py, pgd.py, auto_attack.py) — Person C will import your PGD for adversarial training
- Complete evaluation framework
- All figures and result tables in 05_results/
