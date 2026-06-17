"""
Verify batik ImageFolder dataset structure.

Example:
    python scripts/verify_dataset.py --data_dir data/processed
"""

import argparse
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(class_dir: Path) -> int:
    return sum(1 for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)


def inspect_split(data_dir: Path, split: str):
    split_dir = data_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class folders found under {split_dir}")

    counts = {d.name: count_images(d) for d in class_dirs}
    empty = [cls for cls, count in counts.items() if count == 0]
    if empty:
        raise ValueError(f"Empty class folders in {split}: {empty}")
    return counts


def main():
    parser = argparse.ArgumentParser(description="Verify batik dataset layout")
    parser.add_argument("--data_dir", default="data/processed", help="Processed dataset with train/val folders")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_counts = inspect_split(data_dir, "train")
    val_counts = inspect_split(data_dir, "val")

    train_classes = set(train_counts)
    val_classes = set(val_counts)
    if train_classes != val_classes:
        raise ValueError(
            "Train/val classes differ. "
            f"Only train: {sorted(train_classes - val_classes)}; "
            f"Only val: {sorted(val_classes - train_classes)}"
        )

    classes = sorted(train_classes)
    totals = {cls: train_counts[cls] + val_counts[cls] for cls in classes}
    min_total = min(totals.values())
    max_total = max(totals.values())
    imbalance = max_total / max(1, min_total)

    print(f"✅ Dataset OK: {data_dir}")
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train total: {sum(train_counts.values())}")
    print(f"Val total:   {sum(val_counts.values())}")
    print(f"Imbalance ratio: {imbalance:.2f}x")
    print("\nPer-class counts:")
    for cls in classes:
        print(f"  {cls}: train={train_counts[cls]}, val={val_counts[cls]}, total={totals[cls]}")


if __name__ == "__main__":
    main()
