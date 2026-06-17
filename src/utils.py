import numpy as np
from PIL import Image
import torch
from torchvision import transforms

IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_default_transforms(train: bool = False):
    """Return torchvision transforms for training or validation/inference."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert PIL image to RGB numpy array."""
    return np.array(img.convert("RGB"))


def preprocess_for_model(pil_img: Image.Image, train: bool = False) -> torch.Tensor:
    """Apply standard transforms and return batched tensor (1, C, H, W)."""
    tf = get_default_transforms(train=train)
    tensor = tf(pil_img.convert("RGB")).unsqueeze(0)
    return tensor
