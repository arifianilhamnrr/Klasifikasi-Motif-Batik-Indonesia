"""
Split raw batik dataset folder-per-class into train/val ImageFolder layout.

Example:
    python scripts/prepare_dataset.py --raw_dir data/raw/batik_dataset --out_dir data/processed --val_ratio 0.2
"""

import argparse
import random
import shutil
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(class_dir: Path):
    return sorted([p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTS])


def copy_split(raw_dir: Path, out_dir: Path, val_ratio: float, seed: int):
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw dir not found: {raw_dir}")

    class_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(
            f"No class folders found under {raw_dir}. Expected raw_dir/<class_name>/*.jpg"
        )

    rng = random.Random(seed)
    stats = []

    for class_dir in class_dirs:
        images = list_images(class_dir)
        if not images:
            print(f"[skip] {class_dir.name}: no images")
            continue

        rng.shuffle(images)
        val_count = max(1, int(len(images) * val_ratio)) if len(images) > 1 else 0
        val_images = set(images[:val_count])

        for img in images:
            split = "val" if img in val_images else "train"
            dest = out_dir / split / class_dir.name / img.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                stem = dest.stem
                suffix = dest.suffix
                dest = dest.with_name(f"{stem}_{abs(hash(str(img))) % 100000}{suffix}")
            shutil.copy2(img, dest)

        train_count = len(images) - val_count
        stats.append((class_dir.name, train_count, val_count, len(images)))

    if not stats:
        raise ValueError(f"No images found under {raw_dir}")

    print("Dataset prepared:")
    print(f"  raw_dir: {raw_dir}")
    print(f"  out_dir: {out_dir}")
    print("\nClass counts:")
    for cls, train_count, val_count, total in stats:
        print(f"  {cls}: train={train_count}, val={val_count}, total={total}")


def main():
    parser = argparse.ArgumentParser(description="Prepare batik ImageFolder dataset")
    parser.add_argument("--raw_dir", required=True, help="Raw dataset directory with class subfolders")
    parser.add_argument("--out_dir", default="data/processed", help="Output processed dataset directory")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Delete existing out_dir before preparing")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val_ratio must be >= 0 and < 1")

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)

    copy_split(raw_dir, out_dir, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
