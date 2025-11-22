import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import rasterio
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models
from PIL import Image
import io
import base64
import json

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "MODEL_PATH": "models/model.pth",
    "TILE_SIZE": 512,
    "NUM_CLASSES": 4,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "COLOR_DIFF_THRESHOLD": 30.0,
    "DEFAULT_PIXEL_AREA_M2": 0.25,  # Default if TIF metadata is missing (0.5m GSD)
}

# Colors for Visualization
CLASS_COLORS = {
    0: [0, 0, 0],  # Background
    1: [34, 139, 34],  # Vegetation
    2: [124, 252, 0],  # Farmland
    3: [255, 215, 0],  # Sand
}


# ==========================================
# 2. MODEL DEFINITION
# ==========================================
class VGG16UNet(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(VGG16UNet, self).__init__()
        vgg16 = models.vgg16_bn(weights=None)
        features = list(vgg16.features.children())
        self.enc1 = nn.Sequential(*features[:6])
        self.enc2 = nn.Sequential(*features[6:13])
        self.enc3 = nn.Sequential(*features[13:23])
        self.enc4 = nn.Sequential(*features[23:33])
        self.enc5 = nn.Sequential(*features[33:43])
        self.pool = nn.MaxPool2d(2, 2)
        self.up1 = self.conv_block(512 + 512, 512)
        self.up2 = self.conv_block(512 + 256, 256)
        self.up3 = self.conv_block(256 + 128, 128)
        self.up4 = self.conv_block(128 + 64, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        e4 = self.enc4(p3)
        p4 = self.pool(e4)
        e5 = self.enc5(p4)
        d1 = F.interpolate(e5, scale_factor=2, mode="bilinear", align_corners=True)
        if d1.size()[2:] != e4.size()[2:]:
            d1 = F.interpolate(d1, size=e4.shape[2:], mode="bilinear")
        d1 = torch.cat([d1, e4], dim=1)
        d1 = self.up1(d1)
        d2 = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=True)
        if d2.size()[2:] != e3.size()[2:]:
            d2 = F.interpolate(d2, size=e3.shape[2:], mode="bilinear")
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.up2(d2)
        d3 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=True)
        if d3.size()[2:] != e2.size()[2:]:
            d3 = F.interpolate(d3, size=e2.shape[2:], mode="bilinear")
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.up3(d3)
        d4 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=True)
        if d4.size()[2:] != e1.size()[2:]:
            d4 = F.interpolate(d4, size=e1.shape[2:], mode="bilinear")
        d4 = torch.cat([d4, e1], dim=1)
        d4 = self.up4(d4)
        return self.final_conv(d4)


# ==========================================
# 3. SINGLE INFERENCE ENGINE
# ==========================================
class FloodAnalyzer:
    def __init__(self, config):
        self.config = config
        self.device = config["DEVICE"]

        # Load Model
        self.model = VGG16UNet(num_classes=config["NUM_CLASSES"])
        try:
            state_dict = torch.load(config["MODEL_PATH"], map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e

        self.transform = A.Compose(
            [
                A.Resize(height=config["TILE_SIZE"], width=config["TILE_SIZE"]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def load_image(self, path):
        """Loads TIF or Standard Image and returns (ImageArray, PixelArea)."""
        pixel_area = self.config["DEFAULT_PIXEL_AREA_M2"]
        img_np = None

        if path.lower().endswith((".tif", ".tiff")):
            try:
                with rasterio.open(path) as src:
                    # Read image
                    img_data = src.read()
                    img_np = np.transpose(img_data, (1, 2, 0))  # CHW -> HWC

                    # Try to get area from metadata
                    res_x, res_y = src.res
                    pixel_area = abs(res_x * res_y)

                    # Handle multi-band (take RGB)
                    if img_np.shape[2] > 3:
                        img_np = img_np[:, :, :3]
            except Exception as e:
                print(f"Error reading TIF: {e}")
        else:
            # PNG/JPG
            try:
                img_pil = Image.open(path).convert("RGB")
                img_np = np.array(img_pil)
                # Pixel area remains default since PNG has no geo-metadata
            except Exception as e:
                print(f"Error reading Image: {e}")

        # Resize to model size for consistency
        if img_np is not None:
            img_np = cv2.resize(
                img_np, (self.config["TILE_SIZE"], self.config["TILE_SIZE"])
            )
            img_np = img_np.astype(np.uint8)

        return img_np, pixel_area

    def get_segmentation(self, image_np):
        aug = self.transform(image=image_np)
        img_tensor = aug["image"].unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(img_tensor)
            logits = F.interpolate(
                logits, size=image_np.shape[:2], mode="bilinear", align_corners=False
            )
            pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
        return pred_mask

    def get_lab_diff(self, img_pre, img_post, pre_mask):
        roi_mask = (pre_mask == 1) | (pre_mask == 2)  # Veg or Farm only

        lab_pre = cv2.cvtColor(img_pre, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_post = cv2.cvtColor(img_post, cv2.COLOR_RGB2LAB).astype(np.float32)

        diff = np.sqrt(np.sum((lab_pre - lab_post) ** 2, axis=2))

        masked_diff = np.zeros_like(diff)
        masked_diff[roi_mask] = diff[roi_mask]  # Only calc diff in ROI

        is_damaged = masked_diff > self.config["COLOR_DIFF_THRESHOLD"]
        return is_damaged, masked_diff

    def np_to_base64(self, img_np, cmap=None):
        """Converts Numpy image to Base64 string for Frontend."""
        img_pil = Image.fromarray(img_np.astype("uint8"))

        if cmap == "heatmap":
            # Apply color map for heatmap visualization
            norm_img = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX)
            colored = cv2.applyColorMap(norm_img.astype(np.uint8), cv2.COLORMAP_MAGMA)
            colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(colored)

        buff = io.BytesIO()
        img_pil.save(buff, format="PNG")
        img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"

    def colorize_mask(self, mask):
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, color in CLASS_COLORS.items():
            color_mask[mask == cls_id] = color
        return color_mask

    def create_overlay(self, post_img, is_damaged):
        overlay = post_img.copy()
        # Red overlay on damaged parts
        overlay[is_damaged] = [255, 0, 0]
        # Blend
        final = cv2.addWeighted(overlay, 0.4, post_img, 0.6, 0)
        # Draw contours for sharpness
        contours, _ = cv2.findContours(
            is_damaged.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(final, contours, -1, (255, 0, 0), 2)
        return final

    def process_pair(self, pre_path, post_path):
        """
        MAIN FUNCTION: Takes 2 paths, returns Dictionary result.
        """
        # 1. Load Images
        img_pre, area_per_pixel = self.load_image(pre_path)
        img_post, _ = self.load_image(post_path)

        if img_pre is None or img_post is None:
            return {"error": "Could not load images"}

        # 2. AI Inference (Segmentation)
        pre_mask = self.get_segmentation(img_pre)

        # 3. Change Detection
        is_damaged, diff_map = self.get_lab_diff(img_pre, img_post, pre_mask)

        # 4. Calculate Statistics
        farm_pixels = np.sum(pre_mask == 2)
        veg_pixels = np.sum(pre_mask == 1)

        farm_loss_pixels = np.sum((pre_mask == 2) & is_damaged)
        veg_loss_pixels = np.sum((pre_mask == 1) & is_damaged)

        total_farm_area = farm_pixels * area_per_pixel
        total_veg_area = veg_pixels * area_per_pixel
        lost_farm_area = farm_loss_pixels * area_per_pixel
        lost_veg_area = veg_loss_pixels * area_per_pixel

        # 5. Generate Visuals (Base64)
        visuals = {
            "pre_image": self.np_to_base64(img_pre),
            "post_image": self.np_to_base64(img_post),
            "mask_image": self.np_to_base64(self.colorize_mask(pre_mask)),
            "heatmap_image": self.np_to_base64(diff_map, cmap="heatmap"),
            "overlay_image": self.np_to_base64(
                self.create_overlay(img_post, is_damaged)
            ),
        }

        # 6. Construct Final Response
        response = {
            "status": "success",
            "metadata": {
                "pixel_resolution_m2": round(area_per_pixel, 4),
                "tile_size": self.config["TILE_SIZE"],
            },
            "stats": {
                "farmland": {
                    "total_area_m2": round(total_farm_area, 2),
                    "lost_area_m2": round(lost_farm_area, 2),
                    "percent_loss": round(
                        (lost_farm_area / (total_farm_area + 1e-6)) * 100, 1
                    ),
                },
                "vegetation": {
                    "total_area_m2": round(total_veg_area, 2),
                    "lost_area_m2": round(lost_veg_area, 2),
                    "percent_loss": round(
                        (lost_veg_area / (total_veg_area + 1e-6)) * 100, 1
                    ),
                },
                "total_loss_m2": round(lost_farm_area + lost_veg_area, 2),
            },
            "legend": [
                {"label": "Farmland", "color": "#7CFC00"},  # Hex for Frontend
                {"label": "Vegetation", "color": "#228B22"},
                {"label": "Damage/Loss", "color": "#FF0000"},
            ],
            "images": visuals,
        }

        return response


# ==========================================
# 4. EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    # Initialize Engine
    analyzer = FloodAnalyzer(CONFIG)

    # Define Inputs
    pre_file = "/kaggle/input/data-test/tile_0813.tif"  # Replace with your tile path
    post_file = "/kaggle/input/data-test/tile_0814.tif"  # Replace with your tile path

    # 🔴 RUN INFERENCE
    # Check if files exist just for this demo
    if not os.path.exists(pre_file):
        print("⚠️ Demo file not found. Please set 'pre_file' and 'post_file' paths.")
    else:
        result = analyzer.process_pair(pre_file, post_file)

        # 🟢 PRINT RESULT (Frontend would receive this JSON)
        import json

        print(json.dumps({k: v for k, v in result.items() if k != "images"}, indent=4))
        print(f"Images generated: {list(result['images'].keys())}")

        # Optional: Save base64 images to disk to verify they work
        # for name, b64_str in result['images'].items():
        #     with open(f"{name}.png", "wb") as f:
        #         f.write(base64.b64decode(b64_str.split(",")[1]))


def remove(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} removed successfully.")
    else:
        print(f"{file_path} does not exist.")


def run_agri_model(pre_file, post_file):
    analyzer = FloodAnalyzer(CONFIG)

    # 🔴 RUN INFERENCE
    # Check if files exist just for this demo
    if not os.path.exists(pre_file):
        print("⚠️ Demo file not found. Please set 'pre_file' and 'post_file' paths.")
        return None
    else:
        result = analyzer.process_pair(pre_file, post_file)
        remove(pre_file)
        remove(post_file)
        yield (json.dumps(result) + "\n").encode("utf-8")
