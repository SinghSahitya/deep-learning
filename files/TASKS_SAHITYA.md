# Tasks — Sahitya (Person A)

## Role: Data Pipeline, Baseline Models, Demo, Submission Packaging

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
- Source: https://github.com/yuezunli/celeb-deepfakeforensics
- Preprocessing: extract frames -> MTCNN face crop -> balance classes -> 70/15/15 split

### Code Structure

```
03_code/
+-- src/
|   +-- data/
|   |   +-- dataset.py              # YOU BUILD THIS
|   |   +-- preprocessing.py        # YOU BUILD THIS
|   |   +-- augmentations.py        # YOU BUILD THIS
|   +-- models/
|   |   +-- baseline_cnn.py         # YOU BUILD THIS
|   |   +-- efficientnet_detector.py # YOU BUILD THIS
|   |   +-- frequency_branch.py     # Person C builds this
|   |   +-- multi_domain_model.py   # Person C builds this
|   +-- attacks/                    # Person B builds these
|   +-- losses/                     # Person C builds these
|   +-- training/
|   |   +-- train_baseline.py       # YOU BUILD THIS
|   |   +-- train_adversarial.py    # Person C builds this
|   |   +-- evaluate.py            # Person B builds this
|   +-- utils/
|       +-- metrics.py             # Person B builds this
|       +-- visualization.py       # Person B builds this
|       +-- config.py              # YOU BUILD THIS
+-- scripts/
|   +-- preprocess_celebdf.py       # YOU BUILD THIS
|   +-- train.py                    # YOU BUILD THIS
|   +-- eval.py                     # Person B builds this
|   +-- demo.py                     # YOU BUILD THIS
+-- configs/
    +-- baseline.yaml               # YOU BUILD THIS
    +-- adversarial_training.yaml   # Person C builds this
    +-- multi_domain.yaml           # Person C builds this
```

### Interface Contracts (CRITICAL — follow these so everyone's code integrates)

**Dataset output format:**
```python
# Each item from the dataset must return:
{
    "image": torch.Tensor,    # shape (3, 224, 224), normalized to [0, 1]
    "label": torch.Tensor,    # scalar, 0 = real, 1 = fake
    "path": str               # original file path for debugging
}
```

**Model forward pass contract:**
```python
class Model(nn.Module):
    def forward(self, x):
        # x: (B, 3, 224, 224), values in [0, 1]
        # Returns dict:
        #   "prediction": (B, 1) — sigmoid output
        #   "spatial_features": (B, 256) — features before classifier
        #   "freq_features": (B, 128) or None for baseline models
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

## Your Tasks

---

### TASK 1: Download and Organize Dataset

**What**: Get Celeb-DF v2 and set up the raw data directory.

**Steps**:
1. Go to https://github.com/yuezunli/celeb-deepfakeforensics and fill out the Google Form to request access. The download link is typically emailed within 24 hours. If that's slow, search Kaggle for "Celeb-DF v2".
2. Download both Celeb-real and Celeb-synthesis folders (the YouTube-real folder is optional).
3. Organize as:
   ```
   data/
   +-- celeb_df_raw/
   |   +-- Celeb-real/       # Real celebrity videos (.mp4)
   |   +-- Celeb-synthesis/  # Deepfake videos (.mp4)
   |   +-- List_of_testing_videos.txt  # Official test split
   ```
4. Verify video counts match expected (~600 real, ~5600 fake).

**Output**: Raw video files organized in the directory structure above.

---

### TASK 2: Build Frame Extraction and Face Cropping Pipeline (`preprocessing.py` + `preprocess_celebdf.py`)

**What**: A script that takes raw Celeb-DF videos and produces a directory of cropped face images ready for training.

**File: `src/data/preprocessing.py`** — Core preprocessing functions:

```python
# Functions to implement:

def extract_frames(video_path, output_dir, frame_interval=10):
    """
    Extract every Nth frame from a video using OpenCV.
    - Open video with cv2.VideoCapture
    - Read frames, save every `frame_interval`-th frame as PNG
    - Name format: {video_name}_frame_{frame_number}.png
    - Return list of saved frame paths
    """

def crop_faces(frame_path, output_path, detector):
    """
    Detect and crop the face from a frame using MTCNN.
    - detector: facenet_pytorch.MTCNN instance
    - Detect face bounding box
    - Crop with some margin (20% padding on each side)
    - Resize to 224x224
    - Save to output_path
    - Return True if face found, False otherwise
    """

def balance_dataset(real_paths, fake_paths):
    """
    Balance the dataset by undersampling the majority class.
    - Count real and fake samples
    - Randomly sample from the larger class to match the smaller
    - Return balanced lists
    """

def create_splits(image_paths, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split data into train/val/test using sklearn.model_selection.train_test_split.
    - Stratify by label
    - Save split info as CSV files: train.csv, val.csv, test.csv
    - Each CSV has columns: path, label
    """
```

**File: `scripts/preprocess_celebdf.py`** — Entry point script:
- Parse command-line args: input_dir, output_dir, frame_interval
- Initialize MTCNN detector: `MTCNN(image_size=224, margin=40, keep_all=False, device='cuda')`
- Process all real videos -> extract frames -> crop faces -> save to `data/celeb_df_frames/real/`
- Process all fake videos -> extract frames -> crop faces -> save to `data/celeb_df_frames/fake/`
- Balance the dataset
- Create train/val/test splits, save as CSV
- Print statistics: total frames extracted, faces found, final balanced count per split

**Important details**:
- Use `frame_interval=10` (every 10th frame) to keep dataset manageable. This should give ~15,000-25,000 frames per class after balancing.
- MTCNN from `facenet_pytorch` handles face detection. Install via `pip install facenet-pytorch`.
- Handle videos where no face is detected (skip those frames).
- Set random seed (42) for reproducibility.

**Output**: Directory of 224x224 face images + CSV split files.

---

### TASK 3: Build PyTorch Dataset Class (`dataset.py`)

**What**: A PyTorch Dataset that loads preprocessed face images from CSV split files.

**File: `src/data/dataset.py`**:

```python
class DeepfakeDataset(torch.utils.data.Dataset):
    """
    Args:
        csv_path: path to train.csv / val.csv / test.csv
        transform: torchvision transforms (optional)
        
    Each CSV has columns: path, label
    
    __getitem__ returns:
        {
            "image": torch.Tensor (3, 224, 224) in [0, 1],
            "label": torch.Tensor scalar (0 or 1),
            "path": str
        }
    """
```

Implementation details:
- Load image with PIL, convert to RGB
- Default transform: `Resize(224)` -> `ToTensor()` (which gives [0,1] range)
- Training transform should add: `RandomHorizontalFlip()`, `RandomRotation(10)`, `ColorJitter(brightness=0.2, contrast=0.2)`
- Validation/test transform: just `Resize(224)` -> `ToTensor()`
- Also implement a `get_dataloaders(config)` helper function that returns train/val/test DataLoaders given a config dict

**Output**: Working Dataset class that other team members can import.

---

### TASK 4: Build Data Augmentations (`augmentations.py`)

**What**: Standard image augmentations for training.

**File: `src/data/augmentations.py`**:

```python
def get_train_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),  # Converts to [0, 1]
    ])

def get_val_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
```

Keep it simple. No normalization to ImageNet stats — the attacks and frequency branch need inputs in [0, 1] range. If EfficientNet needs ImageNet normalization, do it inside the model's forward pass, not in the dataset.

---

### TASK 5: Build Baseline CNN Model (`baseline_cnn.py`)

**What**: A simple 4-layer CNN for binary deepfake classification. This serves as the lower baseline.

**File: `src/models/baseline_cnn.py`**:

```python
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
```

The `spatial_features` key is important — Person C's AFS loss will use it.

---

### TASK 6: Build EfficientNet Detector (`efficientnet_detector.py`)

**What**: EfficientNet-B4 based deepfake detector (the spatial branch of our final model).

**File: `src/models/efficientnet_detector.py`**:

```python
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
```

Notes:
- `timm` handles ImageNet normalization internally for EfficientNet, so inputs in [0, 1] are fine.
- Freeze backbone for first 5 epochs, then unfreeze for fine-tuning (implement this in the training script, not here).

---

### TASK 7: Build Config Loader (`utils/config.py`)

**What**: Utility to load YAML config files.

**File: `src/utils/config.py`**:

```python
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
```

Also create the baseline config file:

**File: `configs/baseline.yaml`**:
```yaml
model:
  name: "efficientnet"
  spatial_dim: 256

training:
  epochs: 20
  batch_size: 32
  lr: 0.0001
  weight_decay: 0.00001
  freeze_backbone_epochs: 5

data:
  root: "./data/celeb_df_frames"
  train_csv: "./data/celeb_df_frames/train.csv"
  val_csv: "./data/celeb_df_frames/val.csv"
  test_csv: "./data/celeb_df_frames/test.csv"
  image_size: 224
  num_workers: 4
```

---

### TASK 8: Build Baseline Training Script (`training/train_baseline.py` + `scripts/train.py`)

**What**: Training loop for the baseline models (no adversarial training yet — that's Person C's job).

**File: `src/training/train_baseline.py`**:

```python
def train_baseline(model, train_loader, val_loader, config, device):
    """
    Standard training loop:
    - Optimizer: Adam with LR and weight_decay from config
    - Loss: BCELoss
    - For each epoch:
        1. Train loop: forward, compute BCE loss, backward, step
        2. Val loop: compute val loss and accuracy
        3. Print epoch stats
        4. Save best model checkpoint (by val accuracy)
    - Implement backbone freezing: if epoch < config.training.freeze_backbone_epochs
      and model has a `backbone` attribute, set backbone.requires_grad_(False)
    - Log training loss, val loss, val accuracy per epoch (save to a list/dict)
    - Return training history dict
    """
```

**File: `scripts/train.py`**:

```python
"""
Entry point for training.
Usage: python scripts/train.py --config configs/baseline.yaml

- Parse args (config path, optional: model type override, gpu id)
- Load config
- Create dataset and dataloaders
- Create model based on config.model.name
- Call train_baseline() or train_adversarial() based on config
- Save final model checkpoint
- Save training history as JSON
"""
```

---

### TASK 9: Build Gradio Demo (`scripts/demo.py`)

**What**: Interactive web UI where a user uploads a video or image and gets a real/fake prediction with confidence score.

**File: `scripts/demo.py`**:

```python
import gradio as gr

"""
Demo should:
1. Accept input: video file OR image file
2. If video: extract a few frames (e.g., 16 evenly spaced), crop faces
3. Run each frame through the model
4. Aggregate predictions (majority vote or average confidence)
5. Display:
   - The face crops extracted
   - Per-frame confidence scores
   - Final verdict: REAL or FAKE with overall confidence
   - A note about adversarial robustness (e.g., "This model is trained to resist adversarial attacks")

UI Layout:
- Title: "Robust Deepfake Detector"
- Input: gr.Video() or gr.Image()
- Output: gr.Label() for verdict, gr.Gallery() for extracted faces, gr.Textbox() for details
- Load the best checkpoint (full multi-domain model)

Usage: python scripts/demo.py --checkpoint path/to/best_model.pth --device cuda
Launch: demo.launch(share=True) for Colab compatibility
"""
```

Also optionally add an "adversarial test" mode where the user can toggle adding FGSM noise to the input to show the model's robustness (this would be very impressive in the demo).

---

### TASK 10: Prepare Submission Folders

**What**: Set up the non-code submission folders with required files.

**Files to create**:

1. **`04_data/data_description.md`**: Describe Celeb-DF v2, number of videos, preprocessing steps, train/val/test split sizes, balancing strategy.

2. **`04_data/dataset_links.txt`**: Link to Celeb-DF v2 GitHub page and any mirrors used.

3. **`04_data/sample_inputs/`**: Copy 5 real and 5 fake sample face images here for quick testing.

4. **`06_demo/demo_instructions.md`**: Step-by-step instructions to run the demo in under 5 minutes. Include: install deps, download checkpoint, run command, expected output.

5. **`06_demo/demo_inputs/`**: Include 2-3 short video clips (real + fake) for demo.

6. **`03_code/requirements.txt`**: Full list of pip dependencies with version pins.

7. **`03_code/README.md`**: Setup instructions, training commands, eval commands, demo commands, hardware used.

---

## Dependencies You Need to Install

```bash
pip install torch torchvision timm facenet-pytorch opencv-python gradio pyyaml tqdm pandas scikit-learn matplotlib Pillow
```

---

## What to Deliver to Your Teammates

- **To Person B (Nitin)**: Trained baseline model checkpoints (baseline_cnn.pth, efficientnet.pth) + preprocessed dataset path + working DataLoader
- **To Person C (Vishesh)**: Preprocessed dataset + working DataLoader + EfficientNet model code (they will import and extend it)
