#!/usr/bin/env python3
"""
Script to download YOLOv8 models (nano versions - smallest size)
"""

import os
from pathlib import Path
from ultralytics import YOLO

def download_yolo_models():
    """Download the smallest YOLOv8 models"""

    # Create weights directory if it doesn't exist
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)

    models_to_download = [
        'yolov8n.pt',      # Nano detection model (~6MB)
        'yolov8n-seg.pt',  # Nano segmentation model (~6MB)
    ]

    print("Downloading YOLOv8 nano models (smallest available)...")

    for model_name in models_to_download:
        model_path = weights_dir / model_name

        if model_path.exists():
            print(f"✓ {model_name} already exists, skipping download")
            continue

        try:
            print(f"Downloading {model_name}...")
            # This will download the model if not present
            model = YOLO(model_name)
            print(f"✓ Successfully downloaded {model_name}")
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}")

    print("\nModel download complete!")
    print("Available models:")
    for model_file in weights_dir.glob("*.pt"):
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(".1f")

if __name__ == "__main__":
    download_yolo_models()
