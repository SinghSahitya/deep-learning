from torchvision import transforms

def get_train_transforms(image_size=224):
    """
    Standard training augmentations keeping input in [0, 1] range.
    No ImageNet normalization applied here to accommodate adversarial attacks.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),  # Converts PIL Image (H x W x C) to Tensor (C x H x W) in [0.0, 1.0]
    ])

def get_val_transforms(image_size=224):
    """
    Standard validation transforms keeping input in [0, 1] range.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
