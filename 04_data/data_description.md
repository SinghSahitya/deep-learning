# Dataset Description

## Celeb-DF v2

- **Source**: https://github.com/yuezunli/celeb-deepfakeforensics
- **Type**: Celebrity deepfake video dataset
- **Real videos**: ~600 (Celeb-real)
- **Fake videos**: ~5,639 (Celeb-synthesis)
- **Deepfake method**: Improved face-swapping synthesis

## Preprocessing

1. Frame extraction: every 10th frame from each video (OpenCV)
2. Face detection and cropping: MTCNN (facenet-pytorch), 224x224 with 20% margin
3. Class balancing: undersample majority class (fake) to match minority (real)
4. Split: 70% train / 15% val / 15% test, stratified by label
5. Random seed: 42

## Final Dataset Statistics

[Fill in after preprocessing]

- Train: [N] images ([N/2] real, [N/2] fake)
- Val: [N] images
- Test: [N] images
- Image size: 224x224 RGB
- Value range: [0, 1] (ToTensor normalization)
