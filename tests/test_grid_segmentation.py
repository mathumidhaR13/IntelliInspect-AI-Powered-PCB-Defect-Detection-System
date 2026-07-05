"""
test_grid_segmentation.py
---------------------------
Visual test: loads a real good PCB image, splits it into grid cells,
draws the grid overlay, and saves it as a PNG for visual inspection.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import cv2
from inference.preprocess import load_config, resize_image
from inference.grid_segmentation import split_into_grids, draw_grid_overlay

# Load config
config = load_config(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))

# Load a real good image (raw, not normalized -- we want to SEE it)
image_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "raw_good", "good_001.jpg")
image = cv2.imread(image_path)

if image is None:
    print("❌ Failed to load image. Check the path:", image_path)
else:
    print(f"✅ Loaded image. Original shape: {image.shape}")

    # Resize to standard size first (same as pipeline would do)
    image_resized = resize_image(image, tuple(config["preprocessing"]["image_size"]))

    # Split into grids using config values
    grid_cfg = config["grid_segmentation"]
    grids = split_into_grids(
        image_resized,
        rows=grid_cfg["rows"],
        cols=grid_cfg["cols"],
        overlap_ratio=grid_cfg["overlap_ratio"]
    )

    print(f"✅ Split into {len(grids)} grid cells (expected {grid_cfg['rows']} x {grid_cfg['cols']} = {grid_cfg['rows']*grid_cfg['cols']})")

    # Draw the overlay
    overlay = draw_grid_overlay(image_resized, grids)

    # Save the result so we can visually inspect it
    output_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "grids", "grid_overlay_test.png")
    cv2.imwrite(output_path, overlay)

    print(f"✅ Grid overlay saved to: {output_path}")
    print("   Open this file to visually confirm the grid split looks correct.")