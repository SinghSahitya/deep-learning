# Demo Instructions

## Setup (one-time)

```bash
cd 03_code
pip install -r requirements.txt
```

## Run Demo

```bash
# From inside 03_code/
python scripts/demo.py --checkpoint_dir 05_results/models --device cuda --share

# CPU-only fallback
python scripts/demo.py --checkpoint_dir 05_results/models --device cpu --share
```

The `--share` flag generates a public Gradio link (useful for Colab or remote servers).

## Model Selection

The demo auto-discovers all `.pth` files in the checkpoint directory:
- `best_baseline_cnn.pth` -> Baseline CNN
- `best_efficientnet.pth` -> EfficientNet-B4 (no defense)
- `best_multi_domain.pth` -> Multi-Domain Detector (Ours)

Use the dropdown in the UI to switch between models.

## What the Demo Shows

1. Upload a video or image (sample inputs in `../06_demo/demo_inputs/`)
2. System extracts and crops faces from frames
3. Model predicts Real/Fake with confidence score
4. Results displayed with per-frame breakdown
5. Toggle "Adversarial Robustness Test" to apply FGSM and check if prediction flips

## Expected Runtime

- Image: ~1-2 seconds
- Short video (10s): ~5 seconds

## Sample Inputs

Pre-loaded sample inputs are in `../06_demo/demo_inputs/`:
- `real/` — real face images for testing
- `fake/` — deepfake face images for testing
