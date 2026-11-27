# Pothole Detection and Severity Estimation

A computer-vision project that trains a YOLO11 detector to locate potholes and then assigns each detected pothole a heuristic severity level using its relative image area and an estimated depth map.

> **Note:** Severity scores are relative estimates, not calibrated measurements of real-world pothole dimensions or road safety.

## Project contents

- `images/` and `labels/` — YOLO-format training and validation data.
- `data.yaml` — dataset configuration for Ultralytics.
- `yolot.py` — trains a YOLO11s object detector.
- `testing.py` — runs detection and depth estimation on validation images; writes annotated images and a CSV report to `severity_results/`.
- `pothole.py` — converts source image/annotation files to YOLO-format dataset folders. The original source archive is not included; configure its input paths before running it.

## Requirements

Use Python with pip. Install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

PyTorch installation can vary by operating system and whether you have a supported NVIDIA GPU. If the `torch` install from `requirements.txt` does not match your hardware, install the appropriate PyTorch build first using the official PyTorch installation selector, then install the remaining requirements.

## Train the detector

From the project root, run:

```powershell
python yolot.py
```

This trains for 50 epochs using `data.yaml`. Ultralytics downloads the `yolo11s.pt` base weights when needed and writes training results under `runs/detect/train/`.

## Run detection and severity estimation

After training has produced `runs/detect/train/weights/best.pt`, run:

```powershell
python testing.py
```

The script processes `images/val/`, downloads the Depth Anything V2 Small model from Hugging Face on its first run, and writes annotated images plus `severity_report.csv` to `severity_results/`. A GPU is used for depth estimation when CUDA is available; otherwise inference runs on CPU.

## Prepare a dataset from source annotations

`pothole.py` expects source images and XML-formatted annotations to be available locally. Update `IMG_DIR`, `HTML_DIR`, and `DATASET_DIR` near the top of that script to match your paths, then run:

```powershell
python pothole.py
```

It makes an 80/20 train/validation split (seed 42), copies matched images and labels into `images/` and `labels/`, and writes `data.yaml`.

## Git and large files

Training runs, model weights, Python environments, and caches are excluded from Git. The image and label data currently in this folder are not ignored. Before publishing the repository, confirm you have permission to redistribute the dataset and comply with the applicable licenses for the pretrained weights and Hugging Face model.
