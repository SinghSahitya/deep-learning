# Robust Deepfake Video Detection under Adversarial Conditions

## Setup

```bash
pip install -r requirements.txt
```

## Hardware Used
- GPU: [Fill in, e.g., NVIDIA T4 16GB / A100 40GB]
- RAM: [Fill in]
- Platform: [Fill in, e.g., Google Colab / Kaggle]

## Data Preprocessing

```bash
python scripts/preprocess_celebdf.py --input_dir data/celeb_df_raw --output_dir data/celeb_df_frames --frame_interval 10
```

## Training

```bash
# Baseline (clean training)
python scripts/train.py --config configs/baseline.yaml

# Adversarial training (EfficientNet + AFS)
python scripts/train.py --config configs/adversarial_training.yaml

# Full multi-domain model
python scripts/train.py --config configs/multi_domain.yaml
```

## Evaluation

```bash
python scripts/eval.py --checkpoint checkpoints/multi_domain_full.pth --config configs/multi_domain.yaml --output_dir ../05_results/
```

## Demo

```bash
python scripts/demo.py --checkpoint checkpoints/multi_domain_full.pth --device cuda
```
