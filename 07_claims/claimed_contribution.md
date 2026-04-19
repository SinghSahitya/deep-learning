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
## What Did Not Work
- **Strict GPU Batching**: MTCNN batch processing crashed on frames with no detectable faces; handled via exception catching to prevent pipeline failure.
- **Vanilla Spatial Classifiers**: Single-branch detectors (EfficientNet-only) achieved >95% clean accuracy but collapsed to <5% under PGD attacks, proving baseline spatial features are non-robust.
- **Augmentation-Based Defense**: Attempting to use standard image augmentations (blurring, Gaussian noise) as a defense proved ineffective against iterative adversarial attacks like PGD.
- **Late-Stage Ensembling**: Merging spatial and frequency predictions at the output layer (late fusion) showed poorer robustness compared to our current feature-level concatenation.
- **Global Environment Paths**: Standard `pip`/`conda` paths were missing on the DLAMI; resolved by manual `venv` activation and initialization.

## What We Believe Is Our Contribution
- A dual-branch (spatial + frequency) architecture for adversarially robust deepfake detection
- Frequency Feature Consistency Loss: applying adversarial feature similarity regularization to frequency-domain features
- Empirical evidence that frequency features provide additional resilience against adversarial attacks beyond what spatial-only AFS provides
