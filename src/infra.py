"""
inferencing.py

Single-image inference for the infrastructure segmentation model.

Usage:
    python inferencing.py /path/to/image.tif
    python inferencing.py /path/to/image.jpg
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image

import cv2
import rasterio
import torch
import segmentation_models_pytorch as smp
from io import BytesIO
import base64
from PIL import Image
import numpy as np

import os

CONFIG = {
    "TILE_SIZE": 1024,
    "DEFAULT_PIXEL_AREA_M2": 1.0,
    "ROOT_DIR": Path("Flood-Segmentation-6"),
    "BEST_MODEL_PATH": Path("models/infra.pt"),
}

# Hard-coded class list (from _classes.csv)
CLASS_NAMES = [
    "background",  # 0
    "Bridge",  # 1
    "Building",  # 2
    "Cottage",  # 3
    "Dam",  # 4
    "Haystack",  # 5
    "House",  # 6
    "Irrigation Channel",  # 7
    "Road",  # 8
    "Temple",  # 9
    "Wall",  # 10
    "log",  # 11
]


def load_class_names(classes_csv: Path):
    """Load class names from _classes.csv (same as notebook)."""
    df = pd.read_csv(classes_csv)

    # Try common name columns, else fallback to the first column.
    lower_cols = [c.lower() for c in df.columns]
    col_name = None
    for candidate in ["name", "class", "label"]:
        if candidate in lower_cols:
            col_name = df.columns[lower_cols.index(candidate)]
            break
    if col_name is None:
        col_name = df.columns[0]

    return list(df[col_name].astype(str))


def build_model(num_classes: int, weights_path: Path, device: torch.device):
    """Create UNet(ResNet34) and load trained weights, compatible with PyTorch 2.6+."""
    # 1) Load checkpoint with weights_only=False (new default is True in 2.6)
    try:
        state = torch.load(weights_path, map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch versions don't have weights_only arg
        state = torch.load(weights_path, map_location=device)

    # 2) If someone ever saved the full model (torch.save(model, ...)), handle that
    if isinstance(state, torch.nn.Module):
        model = state.to(device)
        model.eval()
        return model

    # 3) Otherwise, it's some form of state_dict
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=num_classes,
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    return model


def load_image_and_pixel_area(path: Path, config: dict):
    """
    Loads TIFF or standard image and returns:
        img_np_resized (H, W, 3) uint8
        pixel_area_m2 (float)
        jpg_path (Path or None)  - if input was TIFF, saved JPEG path, else None

    Logic mirrors the snippet you provided, plus:
        * for TIFF: saves a JPEG copy next to the source
    """
    pixel_area = config["DEFAULT_PIXEL_AREA_M2"]
    img_np = None
    jpg_path = None

    path_str = str(path)
    tile_size = config["TILE_SIZE"]

    if path_str.lower().endswith((".tif", ".tiff")):
        try:
            with rasterio.open(path_str) as src:

                img_data = src.read()
                img_np = np.transpose(img_data, (1, 2, 0))

                res_x, res_y = src.res
                pixel_area = abs(res_x * res_y)

                if img_np.shape[2] > 3:
                    img_np = img_np[:, :, :3]

            jpg_path = path.with_suffix(".jpg")
            Image.fromarray(img_np.astype(np.uint8)).save(jpg_path, quality=95)

        except Exception as e:
            print(f"[WARN] Error reading TIFF: {e}")
            return None, None, None
    else:
        # JPG/PNG/etc (no geo metadata -> default pixel area)
        try:
            img_pil = Image.open(path_str).convert("RGB")
            img_np = np.array(img_pil)
        except Exception as e:
            print(f"[WARN] Error reading image: {e}")
            return None, None, None

    # Resize to model size
    if img_np is not None:
        img_np = cv2.resize(img_np, (tile_size, tile_size))
        img_np = img_np.astype(np.uint8)

    return img_np, float(pixel_area), jpg_path


def preprocess_image_for_model(img_np: np.ndarray):
    """
    Apply FLAIR-style normalization and convert to torch tensor:
        input:  img_np uint8 [H, W, 3] (0–255, RGB)
        output: tensor [1, 3, H, W] float32
    """
    # FLAIR normalization stats from notebook (0–255 space)
    means = np.array([105.08, 110.87, 101.82], dtype=np.float32)
    stds = np.array([52.17, 45.38, 44.00], dtype=np.float32)

    img = img_np.astype(np.float32)
    img = (img - means) / stds

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


def load_image(pred_np):
    image = Image.fromarray(pred_np.astype(np.uint8))

    # Convert to bytes
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    # Encode as base64
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    # If you want a data URI to use directly in HTML
    return f"data:image/png;base64,{img_base64}"


def load_image_from_path(img_path):

    image = Image.open(img_path)

    # Convert image to bytes
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    # Encode as base64
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    # Optional: data URI for embedding in HTML
    img_data_uri = f"data:image/png;base64,{img_base64}"

    return img_data_uri


@torch.no_grad()
def run_inference_on_image(
    img_path: Path,
    config: dict = CONFIG,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load classes
    # Use hard-coded class list
    class_names = CLASS_NAMES
    num_classes = len(class_names)

    # Build model and load weights
    model = build_model(num_classes, config["BEST_MODEL_PATH"], device)

    # Load image + pixel area + potential jpg output
    img_np, pixel_area_m2, jpg_path = load_image_and_pixel_area(img_path, config)
    if img_np is None:
        print("[ERROR] Could not load image. Exiting.")
        return

    print(f"[INFO] Image: {img_path}")
    if jpg_path is not None:
        print(f"[INFO] TIFF detected -> saved JPEG copy at: {jpg_path}")
    print(f"[INFO] Pixel area (m^2 / pixel): {pixel_area_m2:.6f}")
    print(f"[INFO] Model tile size: {img_np.shape[1]} x {img_np.shape[0]}")

    # Preprocess and move to device
    x = preprocess_image_for_model(img_np).to(device)

    # Forward pass
    logits = model(x)
    preds = torch.argmax(logits, 1)
    pred_np = preds.squeeze(0).cpu().numpy().astype(np.int64)

    height, width = pred_np.shape

    # Per-class area stats
    unique_labels, counts = np.unique(pred_np, return_counts=True)
    total_pixels = pred_np.size

    class_stats = []
    print("\n===== PREDICTION SUMMARY =====")
    for lbl, cnt in zip(unique_labels, counts):
        if lbl < 0 or lbl >= num_classes:
            continue
        name = class_names[lbl]
        frac = cnt / total_pixels
        area_m2 = cnt * pixel_area_m2

        print(
            f"[{lbl:2d}] {name:20s} "
            f"pixels={cnt:8d}  "
            f"frac={frac:6.3f}  "
            f"area_m2={area_m2:12.2f}"
        )

        class_stats.append(
            {
                "class_id": int(lbl),
                "class_name": name,
                "pixel_count": int(cnt),
                "fraction": float(frac),
                "area_m2": float(area_m2),
            }
        )

    # Save raw mask as PNG for visualisation
    out_mask_path = img_path.with_suffix(".pred_mask_ids.png")
    img_data_uri = load_image(pred_np)

    print(f"\n[INFO] Saved predicted mask (class IDs) to: {out_mask_path}")
    print("Prediction mask shape:", pred_np.shape)
    # --------------------------------------------------
    # JSON-style output (API-like)
    # --------------------------------------------------
    result = {
        "original": load_image_from_path(img_path),
        "mask_image": img_data_uri,
        "classes": class_stats,
    }

    # Print JSON to stdout (like an API response)
    # print("\n===== JSON OUTPUT =====")
    # print(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inferencing.py /path/to/image.(tif|tiff|jpg|png)")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"[ERROR] File not found: {img_path}")
        sys.exit(1)

    result = run_inference_on_image(img_path)


def remove(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} removed successfully.")
    else:
        print(f"{file_path} does not exist.")


def run_infra_model(img_path):
    img_path = Path(img_path)
    if not img_path.exists():
        print(f"[ERROR] File not found: {img_path}")
        sys.exit(1)

    result = run_inference_on_image(img_path)
    print(result)
    remove(img_path)
    yield (json.dumps(result) + "\n").encode("utf-8")
