import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from inference.anomaly_inference import (
    load_trained_model,
    predict_single_image,
    visualize_result,
)
from inference.report_generator import generate_report

if len(sys.argv) < 2:
    print("ERROR: Please provide an image path.")
    print('Usage: python tests\\detect_defect.py "path\\to\\your\\image.jpg"')
    sys.exit(1)

input_image_path = sys.argv[1]

if not os.path.exists(input_image_path):
    print("ERROR: File not found: " + input_image_path)
    sys.exit(1)

project_root = os.path.join(os.path.dirname(__file__), "..")
checkpoint_path = os.path.join(
    project_root, "results", "Patchcore", "IntelliInspectPCB",
    "v1", "weights", "lightning", "model.ckpt"
)

input_filename = os.path.splitext(os.path.basename(input_image_path))[0]
output_image_path = os.path.join(
    os.path.dirname(input_image_path), input_filename + "_RESULT.jpg"
)
output_report_path = os.path.join(
    os.path.dirname(input_image_path), input_filename + "_REPORT.html"
)

print("Loading trained model...")
model = load_trained_model(checkpoint_path)

print("Analyzing: " + input_image_path)
result = predict_single_image(model, input_image_path, threshold=0.5)

print("Score: " + format(result["score"], ".4f"))
print("Result: " + result["label"])

board_id = os.path.basename(input_image_path)
visualize_result(input_image_path, result, output_image_path, board_id=board_id)
generate_report(input_image_path, result, output_image_path, output_report_path, board_id=board_id)

print("Marked-up image saved to: " + output_image_path)
print("Full report saved to: " + output_report_path)
