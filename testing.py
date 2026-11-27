import os
import csv
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO

# 1. Paths (update if needed)
WEIGHTS_PATH = r"runs\detect\train\weights\best.pt"
TEST_IMAGES_DIR = r"C:\Users\sghos\Desktop\ppotchoda\images\val"
OUTPUT_DIR = r"severity_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Load your trained YOLO detector and Depth Anything V2
det_model = YOLO(WEIGHTS_PATH)
depth_estimator = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=0 if torch.cuda.is_available() else -1
)

csv_rows = [["image", "pothole_id", "confidence", "area_ratio", "depth_drop", "severity_score", "severity_level"]]

for fname in os.listdir(TEST_IMAGES_DIR):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        continue

    img_path = os.path.join(TEST_IMAGES_DIR, fname)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        continue

    h, w, _ = img_bgr.shape
    total_pixels = h * w
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    # Generate normalized depth map (higher = closer, lower = deeper)
    depth_out = depth_estimator(img_pil)["depth"]
    depth_map = np.array(depth_out.resize((w, h)), dtype=np.float32)
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

    results = det_model(img_bgr, conf=0.25, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    for idx, (box, conf) in enumerate(zip(boxes, confs), start=1):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        box_area = (x2 - x1) * (y2 - y1)
        if box_area <= 0:
            continue

        # 1. Relative Area Score
        area_ratio = box_area / total_pixels
        norm_area = min(1.0, area_ratio * 8.0)

        # 2. Relative Depth Drop (inside box vs. 15px border around box)
        pad = 15
        ox1, oy1 = max(0, x1 - pad), max(0, y1 - pad)
        ox2, oy2 = min(w, x2 + pad), min(h, y2 + pad)

        outer_mask = np.zeros((h, w), dtype=bool)
        outer_mask[oy1:oy2, ox1:ox2] = True
        outer_mask[y1:y2, x1:x2] = False

        inside_depth = depth_map[y1:y2, x1:x2]
        pit_bottom = np.percentile(inside_depth, 10)
        road_surface = np.mean(depth_map[outer_mask]) if np.any(outer_mask) else np.max(inside_depth)
        depth_drop = max(0.0, road_surface - pit_bottom)

        # 3. Combined Severity Score
        severity = (0.4 * norm_area) + (0.6 * depth_drop)

        if severity >= 0.45:
            level, color = "HIGH", (0, 0, 255)
        elif severity >= 0.25:
            level, color = "MEDIUM", (0, 165, 255)
        else:
            level, color = "LOW", (0, 255, 0)

        label = f"{level} ({severity:.2f})"
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_bgr, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        csv_rows.append([fname, idx, round(float(conf), 3), round(area_ratio, 4), round(depth_drop, 4), round(severity, 4), level])

    cv2.imwrite(os.path.join(OUTPUT_DIR, fname), img_bgr)

with open(os.path.join(OUTPUT_DIR, "severity_report.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

print(f"Done! Check '{OUTPUT_DIR}' for annotated images and 'severity_report.csv'.")
