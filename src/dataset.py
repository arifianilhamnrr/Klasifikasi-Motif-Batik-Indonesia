"""
PyTorch Dataset for batik motif classification.
Supports two common formats:
1. Folder-per-class (ImageFolder style): data/processed/train/Parang/*.jpg
2. CSV with columns: path, label (or label_idx) + optional split column.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.utils import get_default_transforms


class BatikDataset(Dataset):
    def __init__(
        self,
        root_dir: Optional[str] = None,
        csv_file: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        transform=None,
        split: Optional[str] = None,
    ):
        """
        Args:
            root_dir: folder containing subfolders named after classes (ImageFolder)
            csv_file: alternative - path to csv with 'image_path' and 'label' columns
            class_names: explicit ordering of classes. If None will infer.
            transform: torchvision/albumentations transform
            split: optional filter (e.g. 'train'/'val'/'test') if csv has 'split' column
        """
        self.transform = transform or get_default_transforms(train=False)
        self.samples: List[Tuple[str, int]] = []
        self.class_names: List[str] = []
        self.class_to_idx: Dict[str, int] = {}

        if root_dir is not None:
            self._load_from_folder(root_dir, split=split, class_names=class_names)
        elif csv_file is not None:
            self._load_from_csv(csv_file, split=split, class_names=class_names)
        else:
            raise ValueError("Provide either root_dir or csv_file")

    def _load_from_folder(self, root_dir: str, split: Optional[str] = None, class_names: Optional[List[str]] = None):
        root = Path(root_dir)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        search_root = root
        if split:
            split_dir = root / split.lower()
            if split_dir.is_dir() and any(split_dir.iterdir()):
                search_root = split_dir

        classes = class_names or sorted([d.name for d in search_root.iterdir() if d.is_dir()])
        if not classes:
            raise ValueError(f"No class folders found under {search_root}")

        self.class_names = list(classes)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}

        for cls in self.class_names:
            cls_dir = search_root / cls
            if not cls_dir.is_dir():
                continue
            for img_path in cls_dir.rglob("*"):
                if img_path.suffix.lower() in exts:
                    self.samples.append((str(img_path), self.class_to_idx[cls]))

        if not self.samples:
            raise ValueError(f"No images found under {search_root}")

    def _load_from_csv(self, csv_file: str, split: Optional[str], class_names: Optional[List[str]]):
        df = pd.read_csv(csv_file)

        if split and "split" in df.columns:
            df = df[df["split"].str.lower() == split.lower()]

        if class_names is None:
            if "label" in df.columns:
                unique = sorted(df["label"].unique().tolist())
            else:
                unique = sorted(df["label_idx"].unique().tolist())
            self.class_names = [str(u) for u in unique]
        else:
            self.class_names = class_names

        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        for _, row in df.iterrows():
            path = row["image_path"] if "image_path" in df.columns else row["path"]
            if "label_idx" in df.columns:
                label = int(row["label_idx"])
            else:
                label = self.class_to_idx[str(row["label"])]
            self.samples.append((str(path), label))

        if not self.samples:
            raise ValueError(f"No samples found in {csv_file} for split={split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_counts(self) -> Dict[str, int]:
        counts = {}
        for _, lbl in self.samples:
            cls = self.class_names[lbl]
            counts[cls] = counts.get(cls, 0) + 1
        return counts


def get_dataloaders(
    data_dir: Optional[str] = None,
    csv_file: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform=None,
    val_transform=None,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Get train/val loaders. Assumes 'train' and 'val' subdirs or csv with split."""
    shared_class_names = class_names
    if shared_class_names is None and data_dir:
        train_root = Path(data_dir) / "train"
        if train_root.is_dir():
            shared_class_names = sorted([d.name for d in train_root.iterdir() if d.is_dir()])

    train_ds = BatikDataset(
        root_dir=data_dir,
        csv_file=csv_file,
        class_names=shared_class_names,
        transform=train_transform or get_default_transforms(train=True),
        split="train",
    )
    val_ds = BatikDataset(
        root_dir=data_dir,
        csv_file=csv_file,
        class_names=shared_class_names,
        transform=val_transform or get_default_transforms(train=False),
        split="val",
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_ds.class_names
