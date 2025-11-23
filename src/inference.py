"""
inferencing.py

Single-image inference for the infrastructure segmentation model.

Usage:
    python inferencing.py /path/to/image.tif
    python inferencing.py /path/to/image.jpg
"""

import base64
from io import BytesIO
import sys
from pathlib import Path
import json
import numpy as np
from PIL import Image
import cv2
import rasterio
import torch
import segmentation_models_pytorch as smp
import os

CONFIG = {
    "TILE_SIZE": 1024,
    "DEFAULT_PIXEL_AREA_M2": 1.0,
    "ROOT_DIR": Path("Flood-Segmentation-6"),
    "BEST_MODEL_PATH": Path("models/infra.pt"),
}

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

REMOVE_CLASSES = [0, 4, 10, 7]  # background, Dam, Wall,Irrigation channel


def build_model(num_classes: int, weights_path: Path, device: torch.device):
    state = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(state, torch.nn.Module):
        model = state.to(device)
        model.eval()
        return model
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
    pixel_area = config["DEFAULT_PIXEL_AREA_M2"]
    img_np = None
    path_str = str(path)
    tile_size = config["TILE_SIZE"]

    if path_str.lower().endswith((".tif", ".tiff")):
        with rasterio.open(path_str) as src:
            img_data = src.read()
            img_np = np.transpose(img_data, (1, 2, 0))
            res_x, res_y = src.res
            pixel_area = abs(res_x * res_y)
            if img_np.shape[2] > 3:
                img_np = img_np[:, :, :3]
    else:
        img_np = np.array(Image.open(path_str).convert("RGB"))

    img_np = cv2.resize(img_np, (tile_size, tile_size)).astype(np.uint8)
    return img_np, float(pixel_area)


def preprocess_image_for_model(img_np: np.ndarray):
    means = np.array([105.08, 110.87, 101.82], dtype=np.float32)
    stds = np.array([52.17, 45.38, 44.00], dtype=np.float32)
    img = (img_np.astype(np.float32) - means) / stds
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


def remove_unwanted_classes(pred_np, remove_ids):
    pred_clean = pred_np.copy()
    for cid in remove_ids:
        pred_clean[pred_clean == cid] = -1
    return pred_clean


def create_color_mask_and_overlay(pred_np: np.ndarray, img_np: np.ndarray, alpha=0.5):
    """
    Creates an overlay image with fixed colors per class.
    Classes marked as -1 (ignored) are transparent.
    """

    # Define fixed BGR colors for each class (index = class id)
    CLASS_COLORS = [
        (0, 0, 0),  # 0: background -> black
        (255, 153, 51),  # 1: Bridge -> bright orange (still "bridge-like")
        (0, 204, 0),  # 2: Building -> bright green
        (
            102,
            255,
            102,
        ),  # 3: Cottage -> light green (matches the idea of cottages/buildings)
        (255, 51, 51),  # 4: Dam -> bright red
        (255, 255, 51),  # 5: Haystack -> bright yellow
        (153, 102, 255),  # 6: House -> vibrant purple
        (51, 102, 255),  # 7: Irrigation Channel -> bright blue
        (255, 0, 0),  # 8: Road -> vivid red
        (204, 0, 204),  # 9: Temple -> bright purple (matches the color family)
        (128, 128, 128),  # 10: Wall -> medium gray
        (0, 153, 153),  # 11: Log -> teal
    ]

    valid_mask = pred_np >= 0
    color_mask = np.zeros((*pred_np.shape, 3), dtype=np.uint8)

    # Apply fixed colors
    for cid, color in enumerate(CLASS_COLORS):
        color_mask[pred_np == cid] = color

    # Overlay on original image
    overlay = img_np.copy()
    overlay[valid_mask] = cv2.addWeighted(
        img_np[valid_mask], 1 - alpha, color_mask[valid_mask], alpha, 0
    )

    return overlay


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
def run_inference_on_image(img_path: Path, config: dict = CONFIG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CLASS_NAMES)
    model = build_model(num_classes, config["BEST_MODEL_PATH"], device)

    img_np, pixel_area_m2 = load_image_and_pixel_area(img_path, config)
    x = preprocess_image_for_model(img_np).to(device)

    logits = model(x)
    preds = torch.argmax(logits, 1)
    pred_np = preds.squeeze(0).cpu().numpy().astype(np.int64)
    pred_np = remove_unwanted_classes(pred_np, REMOVE_CLASSES)

    # Save cleaned mask (ignored classes set to 0)
    mask_to_save = np.where(pred_np < 0, 0, pred_np).astype(np.uint8)
    # mask_path = img_path.with_suffix(".pred_mask_ids.png")
    # Image.fromarray(mask_to_save).save(mask_path)

    mask_img = load_image(mask_to_save)

    # Create overlay image
    overlay = create_color_mask_and_overlay(pred_np, img_np)
    # overlay_path = img_path.with_suffix(".overlay.png")
    # Image.fromarray(overlay).save(overlay_path)
    overlay_img = load_image(overlay)

    # Class-wise statistics
    unique_labels, counts = np.unique(pred_np[pred_np >= 0], return_counts=True)
    total_pixels = pred_np[pred_np >= 0].size
    class_stats = [
        {
            "class_id": int(lbl),
            "class_name": CLASS_NAMES[lbl],
            "pixel_count": int(cnt),
            "fraction": float(cnt / total_pixels),
            "area_m2": float(cnt * pixel_area_m2),
        }
        for lbl, cnt in zip(unique_labels, counts)
    ]

    result = {
        "original": load_image_from_path(img_path),
        "overlay_image": overlay_img,
        "mask_image": mask_img,
        "classes": class_stats,
        "removed_classes": REMOVE_CLASSES,
    }

    # result = {
    #     "image_path": str(img_path),
    #     "width": int(pred_np.shape[1]),
    #     "height": int(pred_np.shape[0]),
    #     "pixel_area_m2": float(pixel_area_m2),
    #     "overlay_png": str(overlay_path),
    #     "mask_id_png": str(mask_path),
    #     "classes": class_stats,
    #     "removed_classes": REMOVE_CLASSES,
    # }

    # print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    img_path = Path(sys.argv[1])
    if not img_path.exists():
        sys.exit(1)
    run_inference_on_image(img_path)


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
