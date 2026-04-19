"""
Image augmentations for training and validation.

Owner: Sahitya

NOTE: No ImageNet normalization here — inputs stay in [0, 1].
EfficientNet handles normalization internally via timm.
"""

from torchvision import transforms


def get_train_transforms(image_size=224):
    """Training augmentations: flip, rotation, color jitter."""
    # TODO
    raise NotImplementedError


def get_val_transforms(image_size=224):
    """Validation/test transforms: resize + to tensor only."""
    # TODO
    raise NotImplementedError
