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
            # Geometric — bikin model gak tergantung orientasi/posisi
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
            # Color — variasi pencahayaan/warna
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
            ),
            # Random apply GaussianBlur — bikin model robust ke blur
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
                p=0.3,
            ),
            # Convert
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            # Dropout di level input — regularisasi ekstra
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
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
