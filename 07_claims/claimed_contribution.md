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
- **Vanilla Spatial Detectors**: Initial single-branch classifiers achieved high clean accuracy but collapsed under adversarial attacks, proving baseline spatial features lack inherent robustness.
- **Traditional Augmentation Defense**: Standard methods like Gaussian blurring and noise injection failed to provide resilience against iterative attacks like PGD.
- **Late-Fusion Interfacing**: Merging spatial and frequency predictions via simple late-stage ensembling was less effective than mid-level feature vector concatenation.
- **Memory-Heavy Backbones**: Heavier models like ResNet-152 caused CUDA Out-Of-Memory (OOM) errors during batched adversarial training, necessitating the pivot to EfficientNet-B4.

## What We Believe Is Our Contribution
- A dual-branch (spatial + frequency) architecture for adversarially robust deepfake detection
- Frequency Feature Consistency Loss: applying adversarial feature similarity regularization to frequency-domain features
- Empirical evidence that frequency features provide additional resilience against adversarial attacks beyond what spatial-only AFS provides
