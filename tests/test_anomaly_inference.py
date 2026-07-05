"""
test_anomaly_inference.py
----------------------------
End-to-end test: load trained model, run inference on one real defect
image, and save a marked-up visualization showing the detected defect.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from inference.anomaly_inference import (
    load_trained_model,
    predict_single_image,
    visualize_result,
)

project_root = os.path.join(os.path.dirname(__file__), "..")

# Path to your trained model checkpoint
checkpoint_path = os.path.join(
    project_root, "results", "Patchcore", "IntelliInspectPCB",
    "v1", "weights", "lightning", "model.ckpt"
)

# Path to a real defect image to test on
test_image_path = os.path.join(
    project_root, "dataset", "test_defect", "defect_001.jpg"
)

# Where to save the marked-up result
output_path = os.path.join(
    project_root, "dataset", "test_defect", "defect_001_RESULT.jpg"
)

print("Loading trained model...")
model = load_trained_model(checkpoint_path)

print(f"Running inference on: {test_image_path}")
result = predict_single_image(model, test_image_path, threshold=0.5)

print(f"Score: {result['score']:.4f}")
print(f"Label: {result['label']}")

print("Generating marked-up visualization...")
visualize_result(test_image_path, result, output_path)

print(f"Done! Result saved to: {output_path}")
print("Open this file to see the defect marked on your image.")