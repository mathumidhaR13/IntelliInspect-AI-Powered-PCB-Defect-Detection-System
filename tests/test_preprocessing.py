"""
test_preprocessing.py
----------------------
Quick manual test: runs preprocess_pipeline() on one real image
from dataset/raw_good and confirms output shape/values look correct.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import cv2
from inference.preprocess import load_config, preprocess_pipeline

# Load config
config = load_config(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))

# Load one real good image
image_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "raw_good", "good_001.jpg")
image = cv2.imread(image_path)

if image is None:
    print("❌ Failed to load image. Check the path:", image_path)
else:
    print(f"✅ Loaded image successfully. Original shape: {image.shape}")

    processed = preprocess_pipeline(image, config)

    print(f"✅ Preprocessing complete. Output shape: {processed.shape}")
    print(f"✅ Output dtype: {processed.dtype}")
    print(f"✅ Value range: min={processed.min():.3f}, max={processed.max():.3f}")