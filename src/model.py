"""
Model definitions for batik motif classification.
- Simple custom CNN (good for from-scratch experiments)
- Transfer learning ResNet18 / ResNet50 (recommended starting point)
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class SimpleCNN(nn.Module):
    """Lightweight CNN for quick experiments or small datasets."""

    def __init__(self, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(
    model_name: str = "resnet18",
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.3,
    device: Optional[torch.device] = None,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Factory to get model.
    model_name: 'resnet18', 'resnet50', 'simple_cnn', or any torchvision resnet variant.
    freeze_backbone: if True, freeze all layers except the final classifier (for small datasets)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "simple_cnn":
        model = SimpleCNN(num_classes=num_classes, dropout=dropout)
    elif model_name.startswith("resnet"):
        if model_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet18(weights=weights)
        elif model_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported resnet variant: {model_name}")
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

        # Freeze backbone for transfer learning on small datasets
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            # Unfreeze last conv block (layer4) + classifier head
            # Layer4 extracts high-level features specific to batik patterns
            for param in model.layer4.parameters():
                param.requires_grad = True
            for param in model.fc.parameters():
                param.requires_grad = True
    else:
        try:
            model = getattr(models, model_name)(weights="IMAGENET1K_V1" if pretrained else None)
            if hasattr(model, "fc"):
                in_features = model.fc.in_features
                model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
            elif hasattr(model, "classifier"):
                last = list(model.classifier.children())[-1]
                if isinstance(last, nn.Linear):
                    model.classifier[-1] = nn.Linear(last.in_features, num_classes)
        except Exception as e:
            raise ValueError(f"Could not create model '{model_name}': {e}")

    model = model.to(device)
    model.eval()
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
