"""
Training script for batik motif classification.
Usage examples:
    python src/train.py --data_dir data/processed --epochs 25 --batch_size 32 --lr 1e-4 --save_path checkpoints/batik_best.pth

Supports folder-per-class or CSV dataset.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.model import get_model, count_parameters
from src.dataset import BatikDataset, get_dataloaders
from src.utils import get_default_transforms


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="Train")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="Val"):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))

    return epoch_loss, epoch_acc, cm


def main():
    parser = argparse.ArgumentParser(description="Train Batik Motif CNN")
    parser.add_argument("--data_dir", type=str, default=None, help="Folder with train/val class subfolders")
    parser.add_argument("--csv_file", type=str, default=None, help="CSV with image_path + label columns")
    parser.add_argument("--model_name", type=str, default="resnet18", help="resnet18 | resnet50 | simple_cnn")
    parser.add_argument("--num_classes", type=int, default=None, help="Optional; inferred from dataset by default")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_path", type=str, default="checkpoints/batik_best.pth")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use ImageNet pretrained backbone")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_tf = get_default_transforms(train=True)
    val_tf = get_default_transforms(train=False)

    if args.data_dir:
        train_loader, val_loader, class_names = get_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_transform=train_tf,
            val_transform=val_tf,
        )
    elif args.csv_file:
        train_ds = BatikDataset(csv_file=args.csv_file, split="train", transform=train_tf)
        val_ds = BatikDataset(csv_file=args.csv_file, split="val", transform=val_tf)
        class_names = train_ds.class_names
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        raise ValueError("Must provide --data_dir or --csv_file")

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    model = get_model(
        model_name=args.model_name,
        num_classes=len(class_names),
        pretrained=args.pretrained,
        device=device,
    )
    print(f"Model params: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = 0.0
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, cm = evaluate(model, val_loader, criterion, device, class_names)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.3f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "model_name": args.model_name,
                "num_classes": len(class_names),
                "class_names": class_names,
            }, args.save_path)
            print(f"✅ Saved best model → {args.save_path} (val_acc={best_acc:.3f})")

    print("\nTraining selesai. Best validation accuracy:", round(best_acc, 4))


if __name__ == "__main__":
    main()
