import os
import glob
import shutil
import random
import xml.etree.ElementTree as ET

# 1. Paste your exact folder paths inside r"..."
IMG_DIR = r"C:\Users\sghos\Downloads\archive\images"
HTML_DIR = r"C:\Users\sghos\Downloads\archive\annotations"
DATASET_DIR = r"C:\Users\sghos\Desktop\ppotchoda"

# Quick check of your input folders
print("Sample files in IMG_DIR:", os.listdir(IMG_DIR)[:3])
print("Sample files in HTML_DIR:", os.listdir(HTML_DIR)[:3])

for split in ["train", "val"]:
    os.makedirs(os.path.join(DATASET_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "labels", split), exist_ok=True)

# Find all annotation files (case-insensitive)
html_files = [
    os.path.join(HTML_DIR, f)
    for f in os.listdir(HTML_DIR)
    if f.lower().endswith((".html", ".xml"))
]
print(f"Total annotation files found: {len(html_files)}")

random.seed(42)
random.shuffle(html_files)

split_idx = int(len(html_files) * 0.8)
splits = {"train": html_files[:split_idx], "val": html_files[split_idx:]}

# Map lowercase base name -> full image filename in IMG_DIR
img_lookup = {}
for fname in os.listdir(IMG_DIR):
    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        stem = os.path.splitext(fname)[0].lower()
        img_lookup[stem] = fname

copied = 0
for split, files in splits.items():
    for html_path in files:
        tree = ET.parse(html_path)
        root = tree.getroot()

        # Match by annotation filename OR <filename> tag inside XML
        html_stem = os.path.splitext(os.path.basename(html_path))[0].lower()
        xml_fname = root.find("filename").text.strip() if root.find("filename") is not None else ""
        xml_stem = os.path.splitext(xml_fname)[0].lower()

        matched_img = img_lookup.get(html_stem) or img_lookup.get(xml_stem)
        if not matched_img:
            continue

        size = root.find("size")
        w = float(size.find("width").text)
        h = float(size.find("height").text)

        yolo_lines = []
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            x_center = ((xmin + xmax) / 2.0) / w
            y_center = ((ymin + ymax) / 2.0) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

        # Copy image and save .txt label
        src_img = os.path.join(IMG_DIR, matched_img)
        dst_img = os.path.join(DATASET_DIR, "images", split, matched_img)
        shutil.copy2(src_img, dst_img)

        txt_filename = os.path.splitext(matched_img)[0] + ".txt"
        dst_txt = os.path.join(DATASET_DIR, "labels", split, txt_filename)
        with open(dst_txt, "w") as f:
            f.write("\n".join(yolo_lines))

        copied += 1

yaml_content = f"""path: {os.path.abspath(DATASET_DIR)}
train: images/train
val: images/val

names:
  0: pothole
"""
with open(os.path.join(DATASET_DIR, "data.yaml"), "w") as f:
    f.write(yaml_content)

print(f"SUCCESS: Copied {copied} image+label pairs into {os.path.abspath(DATASET_DIR)}")
