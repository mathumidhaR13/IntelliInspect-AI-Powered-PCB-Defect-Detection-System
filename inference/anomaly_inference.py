import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import cv2
import numpy as np
from anomalib.models import Patchcore
from anomalib.engine import Engine


def load_trained_model(checkpoint_path):
    model = Patchcore.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def predict_single_image(model, image_path, threshold=0.5):
    engine = Engine(accelerator="auto")
    predictions = engine.predict(model=model, data_path=image_path)
    batch = predictions[0]
    score = float(batch.pred_score[0].item())
    anomaly_map = batch.anomaly_map[0].cpu().numpy()
    pred_mask = batch.pred_mask[0].cpu().numpy()
    label = "FAIL" if score >= threshold else "PASS"
    return {
        "score": score,
        "label": label,
        "anomaly_map": anomaly_map,
        "pred_mask": pred_mask,
    }


def visualize_result(original_image_path, result, output_path, board_id=None):
    if board_id is None:
        board_id = os.path.basename(original_image_path)

    original = cv2.imread(original_image_path)
    h, w = original.shape[:2]

    anomaly_map = cv2.resize(result["anomaly_map"], (w, h))
    pred_mask = cv2.resize(
        result["pred_mask"].astype(np.uint8), (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    normalized = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)
    heatmap_color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    panel1 = original.copy()
    cv2.rectangle(panel1, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(panel1, "Board: " + str(board_id), (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    panel2 = cv2.addWeighted(original, 0.7, heatmap_color, 0.3, 0)
    cv2.rectangle(panel2, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(panel2, "Anomaly Heatmap", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    panel3 = original.copy()
    cv2.drawContours(panel3, contours, -1, (0, 0, 255), 3)
    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)
        cv2.putText(panel3, "DEFECT", (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    label_color = (0, 0, 255) if result["label"] == "FAIL" else (0, 200, 0)
    cv2.rectangle(panel3, (0, 0), (w, 40), (0, 0, 0), -1)
    score_text = result["label"] + " (score: " + format(result["score"], ".3f") + ")"
    cv2.putText(panel3, score_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

    combined = cv2.hconcat([panel1, panel2, panel3])
    cv2.imwrite(output_path, combined)
    return combined
