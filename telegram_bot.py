"""
Telegram bot for batik motif classification.
Send an image to the bot, get back the predicted motif + confidence.

Usage:
    python telegram_bot.py --checkpoint checkpoints/batik_best.pth
"""

import argparse
import asyncio
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import BufferedInputFile
from PIL import Image
import torch

from src.inference import load_model_and_meta, predict_batik

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
DEFAULT_CHECKPOINT = "checkpoints/batik_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
class_names = None


def format_class_name(name: str) -> str:
    """Convert 'batik-megamendung' to 'Batik Megamendung'."""
    return name.replace("-", " ").replace("_", " ").title()


async def cmd_start(message: types.Message):
    await message.reply(
        "👋 *Halo!* Saya bot klasifikasi motif batik.\n\n"
        "Kirim saya gambar batik, nanti saya tebak motifnya apa.\n\n"
        "Kelas yang dikenal:\n"
        + "\n".join(f"• {format_class_name(c)}" for c in class_names),
        parse_mode="Markdown",
    )


async def handle_photo(message: types.Message):
    """Handle photo message — run inference."""
    await message.reply("🔍 Sedang menganalisis gambar...")

    try:
        # Download image from Telegram (aiogram v3 API)
        photo = message.photo[-1]  # largest size
        file = await bot.get_file(photo.file_id)
        byte_stream = await bot.download(file)
        image_data = byte_stream.read()
        pil_image = Image.open(io.BytesIO(image_data))

        # Predict
        result = predict_batik(pil_image, model, class_names, device)

        # Format response
        top_name = format_class_name(result["top_class"])
        top_pct = result["top_confidence"] * 100

        top3_lines = []
        for i, item in enumerate(result["all_probs"][:3], 1):
            name = format_class_name(item["class"])
            pct = item["prob"] * 100
            top3_lines.append(f"  {i}. {name} — {pct:.1f}%")
        top3_text = "\n".join(top3_lines)

        emoji = "🟢" if top_pct > 70 else "🟡" if top_pct > 40 else "🔴"

        reply = (
            f"{emoji} *Prediksi: {top_name}*\n"
            f"Kepercayaan: *{top_pct:.1f}%*\n"
            f"Waktu inference: {result['inference_time_ms']:.0f}ms\n\n"
            f"*Top 3:*\n{top3_text}\n\n"
            f"_Catatan: Model masih dalam tahap training, "
            f"akurasi akan meningkat seiring waktu._"
        )
        await message.reply(reply, parse_mode="Markdown")

    except Exception as e:
        await message.reply(f"❌ Error: {e}")


async def handle_document(message: types.Message):
    """Handle document upload - images or zip files."""
    try:
        file_name = message.document.file_name.lower()

        # Handle zip files (dataset upload)
        if file_name.endswith('.zip'):
            await message.reply("📦 Dataset zip detected, downloading...")

            file = await message.document.get_file()
            byte_stream = await bot.download(file)
            zip_data = byte_stream.read()

            # Save to raw data directory
            save_path = f"/root/batik-cnn-classifier/data/raw/{file_name}"
            with open(save_path, 'wb') as f:
                f.write(zip_data)

            file_size_mb = len(zip_data) / (1024 * 1024)
            await message.reply(
                f"✅ Zip file saved!\n\n"
                f"📁 Location: `{save_path}`\n"
                f"📊 Size: {file_size_mb:.2f} MB\n\n"
                f"Ready to extract and integrate into dataset."
            )
            return

        # Handle images (prediction)
        await message.reply("🔍 Sedang menganalisis gambar...")

        file = await message.document.get_file()
        byte_stream = await bot.download(file)
        image_data = byte_stream.read()
        pil_image = Image.open(io.BytesIO(image_data))

        result = predict_batik(pil_image, model, class_names, device)

        top_name = format_class_name(result["top_class"])
        top_pct = result["top_confidence"] * 100

        top3_lines = []
        for i, item in enumerate(result["all_probs"][:3], 1):
            name = format_class_name(item["class"])
            pct = item["prob"] * 100
            top3_lines.append(f"  {i}. {name} — {pct:.1f}%")
        top3_text = "\n".join(top3_lines)

        emoji = "🟢" if top_pct > 70 else "🟡" if top_pct > 40 else "🔴"

        reply = (
            f"{emoji} *Prediksi: {top_name}*\n"
            f"Kepercayaan: *{top_pct:.1f}%*\n"
            f"Waktu inference: {result['inference_time_ms']:.0f}ms\n\n"
            f"*Top 3:*\n{top3_text}"
        )
        await message.reply(reply, parse_mode="Markdown")

    except Exception as e:
        await message.reply(f"❌ Error: {e}")


async def cmd_reload(message: types.Message):
    """Reload model from checkpoint (useful after training finishes)."""
    global model, class_names
    try:
        model, class_names = load_model_and_meta(
            checkpoint_path=args.checkpoint, device=device
        )
        await message.reply(
            f"✅ Model berhasil di-reload!\n"
            f"Checkpoint: `{args.checkpoint}`\n"
            f"Classes: {len(class_names)}"
        )
    except Exception as e:
        await message.reply(f"❌ Gagal reload model: {e}")


async def cmd_classes(message: types.Message):
    """List all supported classes."""
    lines = [f"• {format_class_name(c)}" for c in class_names]
    await message.reply(
        f"🎨 *{len(class_names)} motif batik:*\n" + "\n".join(lines),
        parse_mode="Markdown",
    )


def main():
    global model, class_names, bot

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    args_main = parser.parse_args()
    global args
    args = args_main

    checkpoint_path = os.path.join(ROOT, args_main.checkpoint)
    print(f"Loading model from: {checkpoint_path}")
    model, class_names = load_model_and_meta(checkpoint_path=checkpoint_path, device=device)
    print(f"Loaded {len(class_names)} classes: {class_names}")

    if not BOT_TOKEN:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN environment variable before running the bot.")

    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()

    dp.message.register(cmd_start, Command("start"))
    dp.message.register(cmd_reload, Command("reload"))
    dp.message.register(cmd_classes, Command("classes"))
    dp.message.register(handle_photo, lambda m: m.photo)
    dp.message.register(handle_document, lambda m: m.document)

    print("🤖 Telegram bot started!")
    asyncio.run(dp.start_polling(bot))


if __name__ == "__main__":
    main()
