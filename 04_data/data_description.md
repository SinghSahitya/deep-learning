# Celeb-DF v2 Dataset

This project utilizes the Celeb-DF v2 dataset.

## Specifications
- **Real Videos**: 590 videos
- **Synthesis (Fake) Videos**: 5639 videos
- **Total Subjects**: 59 unique celebrities

## Preprocessing Pipeline
Since the original videos are in HD `.mp4` format, we implemented a custom preprocessing pipeline:
1. **Frame Extraction**: Every 10th frame was extracted from the source videos.
2. **Face Cropping**: We utilized the PyTorch MTCNN architecture to detect faces, pad the bounding box by 20%, and crop.
3. **Balancing**: Because the synthesis dataset vastly outnumbers the real dataset, we randomly undersampled the fake frames to match the count of the real frames.
4. **Resolution**: Final images are `(224, 224, 3)` RGB PNG files.
5. **Split Layout**: Data is split into `70% Train`, `15% Val`, `15% Test` and documented within CSV mapping files.
