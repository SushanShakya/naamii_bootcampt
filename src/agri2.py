import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import rasterio
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import io
import base64
import json
import segmentation_models_pytorch as smp

# ==========================================
# 1. MODEL REGISTRY
# ==========================================
MODEL_REGISTRY = {
    "b4": {
        "arch": "UnetPlusPlus",
        "encoder": "efficientnet-b4",
        "path": "models/b4.pth",
    },
    "resnet": {
        "arch": "Unet",
        "encoder": "resnet34",
        "path": "models/resnet.pth",
    },
}

# ==========================================
# 2. CONFIGURATION
# ==========================================
CONFIG = {
    "TILE_SIZE": 512,
    "NUM_CLASSES": 4,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "R_IDX": 0,
    "G_IDX": 1,
    "B_IDX": 2,
    "NIR_IDX": 3,
    "NDVI_LOSS_THRESHOLD": 0.15,
    "NDWI_GAIN_THRESHOLD": 0.1,
    "DEFAULT_PIXEL_AREA_M2": 0.25,
}

CLASS_COLORS = {0: [0, 0, 0], 1: [34, 139, 34], 2: [124, 252, 0], 3: [255, 215, 0]}


# ==========================================
# 3. ANALYSIS ENGINE
# ==========================================
class FloodAnalyzer:
    def __init__(self, config, model_selection="b4"):
        """
        Args:
            config (dict): General configuration settings.
            model_selection (str): The key of the model to use (e.g., "resnet", "b4").
                                   This overrides the config if passed.
        """
        self.config = config
        self.device = config["DEVICE"]

        # --- 1. RESOLVE MODEL SETTINGS ---
        # We use the variable passed in arguments
        if model_selection not in MODEL_REGISTRY:
            raise ValueError(
                f"Model '{model_selection}' not found. Options: {list(MODEL_REGISTRY.keys())}"
            )

        model_conf = MODEL_REGISTRY[model_selection]
        self.current_model_name = model_selection

        print(f"\n🚀 Initializing Model: {model_selection.upper()}")
        print(f"   - Architecture: {model_conf['arch']}")
        print(f"   - Encoder:      {model_conf['encoder']}")

        # --- 2. BUILD MODEL ARCHITECTURE ---
        try:
            if model_conf["arch"] == "UnetPlusPlus":
                self.model = smp.UnetPlusPlus(
                    encoder_name=model_conf["encoder"],
                    encoder_weights=None,
                    in_channels=3,
                    classes=config["NUM_CLASSES"],
                )
            elif model_conf["arch"] == "Unet":
                self.model = smp.Unet(
                    encoder_name=model_conf["encoder"],
                    encoder_weights=None,
                    in_channels=3,
                    classes=config["NUM_CLASSES"],
                )

            # --- 3. LOAD WEIGHTS ---
            if os.path.exists(model_conf["path"]):
                state_dict = torch.load(model_conf["path"], map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"✅ Weights loaded: {os.path.basename(model_conf['path'])}")
            else:
                print(f"⚠️ WARNING: Weight file not found at {model_conf['path']}")

            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            print(f"❌ Critical Error loading model: {e}")
            raise e

        self.transform = A.Compose(
            [
                A.Resize(height=config["TILE_SIZE"], width=config["TILE_SIZE"]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def load_multispectral(self, path):
        pixel_area = self.config["DEFAULT_PIXEL_AREA_M2"]
        rgb_np, nir_np = None, None

        if path.lower().endswith((".tif", ".tiff")):
            try:
                with rasterio.open(path) as src:
                    data = src.read()
                    try:
                        r = data[self.config["R_IDX"]]
                        g = data[self.config["G_IDX"]]
                        b = data[self.config["B_IDX"]]
                        rgb_np = np.dstack((r, g, b))
                        if data.shape[0] > 3:
                            nir_np = data[self.config["NIR_IDX"]]
                        else:
                            nir_np = r.astype(np.float32)  # Fallback
                    except IndexError:
                        return None, None, None
                    res_x, res_y = src.res
                    pixel_area = abs(res_x * res_y)
            except Exception as e:
                print(f"Error reading TIF: {e}")
        else:
            # Fallback
            try:
                img_pil = Image.open(path).convert("RGB")
                rgb_np = np.array(img_pil)
                nir_np = np.zeros((rgb_np.shape[0], rgb_np.shape[1]))
            except Exception:
                pass

        if rgb_np is not None:
            rgb_np = cv2.resize(
                rgb_np, (self.config["TILE_SIZE"], self.config["TILE_SIZE"])
            ).astype(np.uint8)
        if nir_np is not None:
            nir_np = cv2.resize(
                nir_np, (self.config["TILE_SIZE"], self.config["TILE_SIZE"])
            ).astype(np.float32)

        return rgb_np, nir_np, pixel_area

    def get_segmentation(self, rgb_np):
        aug = self.transform(image=rgb_np)
        img_tensor = aug["image"].unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(img_tensor)
            if logits.shape[-2:] != rgb_np.shape[:2]:
                logits = F.interpolate(
                    logits, size=rgb_np.shape[:2], mode="bilinear", align_corners=False
                )
            pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
        return pred_mask

    def calculate_spectral_indices(self, rgb_np, nir_np):
        rgb_norm = rgb_np.astype(np.float32) / 255.0
        max_nir = np.max(nir_np)
        nir_norm = nir_np / 65535.0 if max_nir > 255 else nir_np / 255.0
        R = rgb_norm[:, :, 0]
        G = rgb_norm[:, :, 1]
        NIR = nir_norm
        ndvi = (NIR - R) / (NIR + R + 1e-6)
        ndwi = (G - NIR) / (G + NIR + 1e-6)
        return ndvi, ndwi

    def detect_damage_nir(self, rgb_pre, nir_pre, rgb_post, nir_post, pre_mask):
        roi_mask = (pre_mask == 1) | (pre_mask == 2)
        ndvi_pre, ndwi_pre = self.calculate_spectral_indices(rgb_pre, nir_pre)
        ndvi_post, ndwi_post = self.calculate_spectral_indices(rgb_post, nir_post)

        ndvi_loss = ndvi_pre - ndvi_post
        loss_by_death = ndvi_loss > self.config["NDVI_LOSS_THRESHOLD"]
        ndwi_gain = ndwi_post - ndwi_pre
        loss_by_water = ndwi_gain > self.config["NDWI_GAIN_THRESHOLD"]

        combined_damage = loss_by_death | loss_by_water
        final_damage = np.zeros_like(combined_damage, dtype=bool)
        final_damage[roi_mask] = combined_damage[roi_mask]
        heatmap = np.zeros_like(ndvi_loss)
        heatmap[roi_mask] = ndvi_loss[roi_mask]
        return final_damage, heatmap

    def colorize_mask(self, mask):
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for c, color in CLASS_COLORS.items():
            rgb[mask == c] = color
        return rgb

    def create_overlay(self, post_img, is_damaged):
        overlay = post_img.copy()
        overlay[is_damaged] = [255, 0, 0]
        final = cv2.addWeighted(overlay, 0.4, post_img, 0.6, 0)
        contours, _ = cv2.findContours(
            is_damaged.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(final, contours, -1, (255, 0, 0), 2)
        return final

    def np_to_base64(self, img_np, cmap=None):
        img_pil = Image.fromarray(img_np.astype("uint8"))
        if cmap == "heatmap":
            norm = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX)
            colored = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
            img_pil = Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
        buff = io.BytesIO()
        img_pil.save(buff, format="PNG")
        return (
            f"data:image/png;base64,{base64.b64encode(buff.getvalue()).decode('utf-8')}"
        )

    def process_pair(self, pre_path, post_path):
        pre_rgb, pre_nir, area_px = self.load_multispectral(pre_path)
        post_rgb, post_nir, _ = self.load_multispectral(post_path)
        if pre_rgb is None or post_rgb is None:
            return {"error": "Failed to load images"}

        pre_mask = self.get_segmentation(pre_rgb)
        is_damaged, loss_heatmap = self.detect_damage_nir(
            pre_rgb, pre_nir, post_rgb, post_nir, pre_mask
        )

        farm_loss = np.sum((pre_mask == 2) & is_damaged)
        veg_loss = np.sum((pre_mask == 1) & is_damaged)

        visuals = {
            "pre_image": self.np_to_base64(pre_rgb),
            "post_image": self.np_to_base64(post_rgb),
            "mask_image": self.np_to_base64(self.colorize_mask(pre_mask)),
            "heatmap_image": self.np_to_base64(loss_heatmap, cmap="heatmap"),
            "overlay_image": self.np_to_base64(
                self.create_overlay(post_rgb, is_damaged)
            ),
        }
        self.last_run = {
            "pre": pre_rgb,
            "post": post_rgb,
            "mask": self.colorize_mask(pre_mask),
            "heatmap": loss_heatmap,
            "overlay": self.create_overlay(post_rgb, is_damaged),
        }
        return {
            "status": "success",
            "metadata": {
                "pixel_resolution_m2": round(area_px, 4),
                "model_used": self.current_model_name,
            },
            "stats": {
                "farmland": {
                    "total_area_m2": round(np.sum(pre_mask == 2) * area_px, 2),
                    "lost_area_m2": round(farm_loss * area_px, 2),
                },
                "vegetation": {
                    "total_area_m2": round(np.sum(pre_mask == 1) * area_px, 2),
                    "lost_area_m2": round(veg_loss * area_px, 2),
                },
                "total_loss_m2": round((farm_loss + veg_loss) * area_px, 2),
            },
            "images": visuals,
        }

    def display_results(self):
        if not hasattr(self, "last_run"):
            return
        imgs = self.last_run
        fig, ax = plt.subplots(1, 5, figsize=(20, 5))
        ax[0].imshow(imgs["pre"])
        ax[0].set_title("Pre (RGB)")
        ax[1].imshow(imgs["post"])
        ax[1].set_title("Post (RGB)")
        ax[2].imshow(imgs["mask"])
        ax[2].set_title(f"Mask ({self.current_model_name})")
        norm = cv2.normalize(imgs["heatmap"], None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        hm = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        ax[3].imshow(cv2.cvtColor(hm, cv2.COLOR_BGR2RGB))
        ax[3].set_title("Loss Heatmap")
        ax[4].imshow(imgs["overlay"])
        ax[4].set_title("Damage Overlay")
        for a in ax:
            a.axis("off")
        plt.show()


# ==========================================
# 4. RUN: CHANGE THE VARIABLE HERE!
# ==========================================
if __name__ == "__main__":
    pre_file = "/content/tile_0813.tif"
    post_file = "/content/tile_0814.tif"

    # -------------------------------------------------------
    # 👇 CHANGE THIS VARIABLE TO "b4" OR "resnet" TO SWITCH
    # -------------------------------------------------------
    MY_MODEL_CHOICE = "resnet"
    # -------------------------------------------------------

    if os.path.exists(pre_file):
        print(f"--- STARTING ANALYSIS WITH MODEL: {MY_MODEL_CHOICE} ---")

        # Pass the variable directly here
        analyzer = FloodAnalyzer(CONFIG, model_selection=MY_MODEL_CHOICE)

        result = analyzer.process_pair(pre_file, post_file)
        print(json.dumps(result["stats"], indent=4))
        analyzer.display_results()
    else:
        print("Please ensure TIF files exist.")


def run_agri2(pre_file, post_file, model):
    MY_MODEL_CHOICE = model
    if os.path.exists(pre_file):
        print(f"--- STARTING ANALYSIS WITH MODEL: {MY_MODEL_CHOICE} ---")

        # Pass the variable directly here
        analyzer = FloodAnalyzer(CONFIG, model_selection=MY_MODEL_CHOICE)

        result = analyzer.process_pair(pre_file, post_file)

        yield (json.dumps(result) + "\n").encode("utf-8")
    else:
        print("Please ensure TIF files exist.")
