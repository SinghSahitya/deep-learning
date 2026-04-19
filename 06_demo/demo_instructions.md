# Demo Instructions

## Setup (one-time)

```bash
cd 03_code
pip install -r requirements.txt
```

## Download Checkpoint

[Fill in: link to checkpoint or path]

## Run Demo

```bash
python scripts/demo.py --checkpoint checkpoints/multi_domain_full.pth --device cuda --share
```

## What the Demo Shows

1. Upload a video or image
2. System extracts faces from frames
3. Model predicts Real/Fake with confidence score
4. Results displayed with per-frame breakdown

## Expected Runtime

- Image: ~1 second
- Short video (10s): ~5 seconds

## Backup

If live demo fails, see backup_video.mp4 for a recorded demonstration.
