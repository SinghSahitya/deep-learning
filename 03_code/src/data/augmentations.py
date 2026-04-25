"""
Image augmentations for training and validation.

Owner: Sahitya

NOTE: No ImageNet normalization here — inputs stay in [0, 1].
EfficientNet handles normalization internally via timm.
"""

import random
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image


def get_train_transforms(image_size=224):
    """Training augmentations: flip, rotation, color jitter."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1
            ),
            transforms.ToTensor(),
        ]
    )


def get_val_transforms(image_size=224):
    """Validation/test transforms: resize + to tensor only."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


class ClipTrainTransform:
    """Apply identical random augmentation to every frame in a clip."""

    def __init__(self, image_size=224):
        self.image_size = image_size

    def __call__(self, frames):
        """
        Args:
            frames: list of PIL Images (T frames from one clip)
        Returns:
            torch.Tensor of shape (T, 3, image_size, image_size)
        """
        do_flip = random.random() < 0.5
        angle = random.uniform(-10, 10)
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        saturation = random.uniform(0.9, 1.1)

        tensors = []
        for img in frames:
            img = TF.resize(img, [self.image_size, self.image_size])
            if do_flip:
                img = TF.hflip(img)
            img = TF.rotate(img, angle)
            img = TF.adjust_brightness(img, brightness)
            img = TF.adjust_contrast(img, contrast)
            img = TF.adjust_saturation(img, saturation)
            tensors.append(TF.to_tensor(img))

        return torch.stack(tensors, dim=0)


class ClipValTransform:
    """Resize + to tensor for every frame in a clip (no randomness)."""

    def __init__(self, image_size=224):
        self.image_size = image_size

    def __call__(self, frames):
        """
        Args:
            frames: list of PIL Images
        Returns:
            torch.Tensor of shape (T, 3, image_size, image_size)
        """
        tensors = []
        for img in frames:
            img = TF.resize(img, [self.image_size, self.image_size])
            tensors.append(TF.to_tensor(img))
        return torch.stack(tensors, dim=0)
