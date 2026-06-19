"""
Training script for batik motif classification.
Usage examples:
    python src/train.py --data_dir data/processed --epochs 25 --batch_size 32 --lr 1e-4 --save_path checkpoints/batik_best.pth

Supports folder-per-class or CSV dataset.
"""

import argparse
import json
import os
import signal
import sys
import urllib.request
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


TELEGRAM_BOT_TOKEN = "8866545134:AAHZBTc78xP9jlRCT4wmW4vGgyvdoumJ30c"
TELEGRAM_CHAT_ID = "6215646464"

# Global state for crash notification
_training_state = {
    "epoch": 0,
    "total_epochs": 0,
    "best_acc": 0.0,
    "model_name": "",
    "started": False,
}


def send_telegram_notification(message):
    """Send notification to Telegram (HTML format)."""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = json.dumps({
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[Telegram] Failed to send notification: {e}")


def _crash_handler(signum=None, frame=None, error=None):
    """Send crash notification to Telegram."""
    reason = "Unknown"
    if error:
        reason = f"{type(error).__name__}: {error}"
    elif signum is not None:
        sig_name = signal.Signals(signum).name if signum else "SIGTERM"
        reason = f"Signal {sig_name}"

    best_pct = _training_state['best_acc'] * 100
    msg = (
        f"🚨 <b>Training Terhenti!</b>\n\n"
        f"❌ Alasan: <code>{reason}</code>\n"
        f"Model: <code>{_training_state['model_name']}</code>\n"
        f"Epoch terakhir: <b>{_training_state['epoch']}/{_training_state['total_epochs']}</b>\n"
        f"🏆 Akurasi terbaik: <b>{best_pct:.1f}%</b>"
    )
    send_telegram_notification(msg)


def _signal_handler(signum, frame):
    _crash_handler(signum=signum)
    sys.exit(1)


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


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

    # Generate per-class metrics for detailed reporting
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    per_class_metrics = {
        class_names[i]: {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        }
        for i in range(len(class_names))
    }

    return epoch_loss, epoch_acc, cm, per_class_metrics


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
    parser.add_argument("--freeze_backbone", action="store_true", default=False, help="Freeze backbone layers, train only classifier head")
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
        freeze_backbone=args.freeze_backbone,
    )

    if args.freeze_backbone:
        print(f"Frozen backbone mode: training only classifier head")
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable_params = list(model.parameters())

    print(f"Model params: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler - cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Early stopping parameters
    patience = 10  # stop if no improvement for 10 epochs
    epochs_no_improve = 0

    best_acc = 0.0
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    _training_state.update({
        "total_epochs": args.epochs,
        "model_name": args.model_name,
        "started": True,
    })

    try:
        for epoch in range(1, args.epochs + 1):
            _training_state["epoch"] = epoch

            print(f"\n=== Epoch {epoch}/{args.epochs} ===")
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, cm, per_class_metrics = evaluate(model, val_loader, criterion, device, class_names)

            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.3f}")
            print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.3f}")

            if val_acc > best_acc:
                best_acc = val_acc
                _training_state["best_acc"] = best_acc
                epochs_no_improve = 0
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
            else:
                epochs_no_improve += 1
                print(f"⚠️ No improvement for {epochs_no_improve} epoch(s)")

            # Step the scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"📉 Learning rate: {current_lr:.6f}")

            # Early stopping check
            if epochs_no_improve >= patience:
                print(f"🛑 Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs)")
                break

            # Telegram notification
            train_pct = train_acc * 100
            val_pct = val_acc * 100
            best_pct = best_acc * 100

            # Progress indicator
            progress = "🟢" if val_pct > 70 else "🟡" if val_pct > 50 else "🔴"

            msg = (
                f"🎨 <b>Update Training - Epoch {epoch}/{args.epochs}</b>\n\n"
                f"📚 <b>Fase Belajar</b>\n"
                f"Akurasi: <b>{train_pct:.1f}%</b> (dari 100%)\n"
                f"<i>Semakin kecil loss ({train_loss:.2f}), semakin bagus</i>\n\n"
                f"🎯 <b>Fase Ujian</b>\n"
                f"Akurasi: <b>{val_pct:.1f}%</b> (dari 100%)\n"
                f"<i>Loss: {val_loss:.2f}</i>\n\n"
                f"🏆 <b>Rekor Terbaik</b>: {best_pct:.1f}%\n"
                f"📉 <b>LR</b>: {current_lr:.6f}\n\n"
                f"{progress} <i>Epoch {epoch} selesai</i>"
            )
            send_telegram_notification(msg)

            # Send detailed per-class metrics in separate message
            detail_msg = f"📋 <b>Detail Per Kelas - Epoch {epoch}</b>\n\n"
            for class_name, metrics in per_class_metrics.items():
                detail_msg += (
                    f"<b>{class_name}</b> (n={metrics['support']})\n"
                    f"  Prec: {metrics['precision']*100:.1f}% | "
                    f"Rec: {metrics['recall']*100:.1f}% | "
                    f"F1: {metrics['f1']*100:.1f}%\n"
                )
            send_telegram_notification(detail_msg)

    except Exception as e:
        _crash_handler(error=e)
        raise

    # Final notification
    final_pct = best_acc * 100
    final_msg = (
        f"✅ <b>Training Selesai!</b>\n\n"
        f"🎉 <b>Hasil Akhir</b>\n"
        f"Model: <code>{args.model_name}</code>\n"
        f"Total Epoch: <b>{args.epochs}</b>\n"
        f"🏆 Akurasi Terbaik: <b>{final_pct:.1f}%</b>\n\n"
        f"💾 Model tersimpan di:\n<code>{args.save_path}</code>\n\n"
        f"<i>Siap dipakai untuk klasifikasi batik!</i>"
    )
    send_telegram_notification(final_msg)

    print("\nTraining selesai. Best validation accuracy:", round(best_acc, 4))


if __name__ == "__main__":
    main()
