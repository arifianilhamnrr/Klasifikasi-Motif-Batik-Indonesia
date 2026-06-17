"""
Inference pipeline for batik motif classification.
Includes:
- Model loading (supports custom checkpoint)
- Preprocessing
- Prediction with probabilities
- Dummy model creation for demo / pipeline testing
"""

import os
import time
from typing import Dict, List, Tuple, Optional

from PIL import Image
import torch
import torch.nn.functional as F

from src.model import get_model
from src.utils import preprocess_for_model

DEFAULT_CLASS_NAMES = ["Parang", "Kawung", "Mega Mendung", "Truntum", "Sidomukti"]


def create_dummy_checkpoint(
    num_classes: int = 5,
    model_name: str = "resnet18",
    save_path: Optional[str] = None,
    seed: int = 42,
) -> str:
    """
    Create and optionally save a dummy (randomly initialized) model checkpoint.
    Useful for testing the full pipeline + Gradio app immediately.
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = get_model(model_name=model_name, num_classes=num_classes, pretrained=False, device=device)

    class_names = DEFAULT_CLASS_NAMES[:num_classes]
    if len(class_names) < num_classes:
        class_names += [f"Class_{i+1}" for i in range(len(class_names), num_classes)]

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "num_classes": num_classes,
            "class_names": class_names,
        }, save_path)
        print(f"Dummy checkpoint saved to: {save_path}")
    return save_path


def load_model_and_meta(
    checkpoint_path: Optional[str] = None,
    model_name: str = "resnet18",
    num_classes: int = 5,
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, List[str]]:
    """
    Load a trained checkpoint or fall back to a randomly initialized model.
    Returns (model, class_names).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"[inference] Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model_name_ckpt = ckpt.get("model_name", model_name)
        num_classes_ckpt = ckpt.get("num_classes", num_classes)
        class_names = ckpt.get("class_names", DEFAULT_CLASS_NAMES[:num_classes_ckpt])

        model = get_model(
            model_name=model_name_ckpt,
            num_classes=num_classes_ckpt,
            pretrained=False,
            device=device,
        )
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        model.eval()
        return model, class_names

    print("[inference] No valid checkpoint found. Using randomly initialized model (DEMO MODE).")
    model = get_model(model_name=model_name, num_classes=num_classes, pretrained=False, device=device)
    return model, DEFAULT_CLASS_NAMES[:num_classes]


def predict_batik(
    pil_image: Image.Image,
    model: torch.nn.Module,
    class_names: List[str],
    device: Optional[torch.device] = None,
    return_top_k: int = 3,
) -> Dict:
    """
    Full prediction pipeline for batik image.
    Returns dict with top_class, top_confidence, all_probs, full_probs, inference_time_ms.
    """
    if device is None:
        device = next(model.parameters()).device

    start = time.time()
    image = pil_image.convert("RGB")
    input_tensor = preprocess_for_model(image).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    elapsed = (time.time() - start) * 1000

    prob_list = [
        {"class": cls, "prob": float(p)}
        for cls, p in zip(class_names, probs)
    ]
    prob_list.sort(key=lambda x: x["prob"], reverse=True)
    top = prob_list[0]

    return {
        "top_class": top["class"],
        "top_confidence": top["prob"],
        "all_probs": prob_list[:return_top_k],
        "full_probs": prob_list,
        "processed_image": image,
        "inference_time_ms": round(elapsed, 1),
        "num_classes": len(class_names),
    }


def format_results_for_ui(result: Dict) -> Dict:
    """Convert raw prediction result to display-friendly dict."""
    return {
        "prediction": f"{result['top_class']} ({result['top_confidence'] * 100:.1f}%)",
        "confidence": result["top_confidence"],
        "probabilities": {
            item["class"]: round(item["prob"] * 100, 2)
            for item in result["full_probs"]
        },
        "top_k": [
            f"{item['class']}: {item['prob'] * 100:.1f}%"
            for item in result["all_probs"]
        ],
        "processed_image": result.get("processed_image"),
        "inference_ms": result["inference_time_ms"],
    }
