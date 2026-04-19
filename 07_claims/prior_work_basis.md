# Prior Work Basis

## Paper 1: Khan et al. (2024) — Adversarial Feature Similarity Learning
- **Citation**: Khan et al., "Adversarially Robust Deepfake Detection via Adversarial Feature Similarity Learning," arXiv 2403.08806, 2024.
- **Influence**: We reproduced their Adversarial Feature Similarity (AFS) loss and applied it to our EfficientNet-B4 backbone. Our implementation follows their formulation: L_AFS = mean(||f(x) - f(x_adv)||_2).

## Paper 2: Yan et al. (2024) — Latent Space Augmentation (CVPR 2024)
- **Citation**: Yan et al., "Transcending Forgery Specificity with Latent Space Augmentation for Generalizable Deepfake Detection," CVPR 2024.
- **Influence**: Motivated our approach to learning forgery-agnostic features. We did not directly implement their augmentation method, but their insight about generalization influenced our multi-domain feature design.

## Paper 3: Exposing DeepFakes via Hyperspectral Domain Mapping
- **Citation**: arXiv 2511.11732
- **Influence**: Motivated our frequency-domain branch. Their observation that frequency-domain inconsistencies are robust to common perturbations led us to incorporate FFT-based feature extraction.
