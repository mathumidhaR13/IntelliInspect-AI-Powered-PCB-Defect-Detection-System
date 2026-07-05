"""
report_generator.py
----------------------
Generates a structured, human-readable HTML inspection report for a single
board, using REAL data computed by the PatchCore model -- not mock data.

This directly implements the project's "Dashboard" requirement: showing
board info, analysis summary, defect details, and an actionable
recommendation, in a format any non-technical viewer (QA manager, line
supervisor) can understand at a glance.
"""

import os
import cv2
import base64
from datetime import datetime


def _severity_from_score(score):
    """
    Maps a raw anomaly score (0-1) to a human-readable severity level.
    Thresholds are configurable design choices, not fixed constants --
    in production these would be tuned using labeled validation data.
    """
    if score >= 0.75:
        return "High", "#d32f2f"
    elif score >= 0.4:
        return "Medium", "#f57c00"
    else:
        return "Low", "#388e3c"


def _recommendation_for(label, severity):
    if label == "PASS":
        return "No action needed. Board meets quality standards."
    if severity == "High":
        return "Stop this unit. Isolate and inspect immediately before it proceeds further down the line."
    if severity == "Medium":
        return "Flag this unit for manual review by a QA technician before release."
    return "Log for trend monitoring. No immediate action required, but track if this recurs."


def generate_report(original_image_path, result, marked_image_path, output_html_path, board_id=None):
    """
    Builds a full HTML inspection report combining real detection results
    with a clear, structured layout.

    Args:
        original_image_path (str): path to the original uploaded image
        result (dict): output from predict_single_image()
        marked_image_path (str): path to the already-generated marked-up image
        output_html_path (str): where to save the HTML report
        board_id (str, optional): board identifier
    """
    if board_id is None:
        board_id = os.path.basename(original_image_path)

    original = cv2.imread(original_image_path)
    h, w = original.shape[:2]

    severity, severity_color = _severity_from_score(result["score"])
    recommendation = _recommendation_for(result["label"], severity)
    status_color = "#d32f2f" if result["label"] == "FAIL" else "#388e3c"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Embed the marked-up image directly into the HTML as base64,
    # so the report is a single self-contained file (no broken image links).
    with open(marked_image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Inspection Report - {board_id}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #eef1f5; margin: 0; padding: 30px; }}
  .container {{ max-width: 950px; margin: auto; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
  .header {{ background: #1a2a4a; color: white; padding: 18px 25px; font-size: 20px; font-weight: 600; }}
  .image-section {{ padding: 20px; text-align: center; background: #fafafa; }}
  .image-section img {{ max-width: 100%; border-radius: 6px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: #ddd; }}
  .panel {{ background: white; padding: 20px; }}
  .panel h3 {{ margin-top: 0; color: #1a2a4a; border-bottom: 2px solid #eee; padding-bottom: 8px; }}
  .row {{ display: flex; justify-content: space-between; padding: 6px 0; font-size: 14px; }}
  .label {{ color: #666; }}
  .value {{ font-weight: 600; color: #222; }}
  .status-badge {{ display: inline-block; padding: 4px 14px; border-radius: 20px; color: white; font-weight: 700; background: {status_color}; }}
  .recommendation {{ background: #1a2a4a; color: white; padding: 20px 25px; font-size: 15px; }}
  .severity {{ color: {severity_color}; font-weight: 700; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">IntelliInspect &mdash; PCB Anomaly Inspection Report</div>

  <div class="image-section">
    <img src="data:image/jpeg;base64,{img_b64}" alt="Inspection Result">
  </div>

  <div class="grid">
    <div class="panel">
      <h3>Board Information</h3>
      <div class="row"><span class="label">Board ID</span><span class="value">{board_id}</span></div>
      <div class="row"><span class="label">Inspection Time</span><span class="value">{timestamp}</span></div>
      <div class="row"><span class="label">Image Resolution</span><span class="value">{w} x {h}</span></div>
      <div class="row"><span class="label">Model</span><span class="value">PatchCore (ResNet18 backbone)</span></div>
    </div>

    <div class="panel">
      <h3>Analysis Summary</h3>
      <div class="row"><span class="label">Inspection Status</span><span class="value"><span class="status-badge">{result['label']}</span></span></div>
      <div class="row"><span class="label">Anomaly Score</span><span class="value">{result['score']:.3f}</span></div>
      <div class="row"><span class="label">Severity</span><span class="value severity">{severity}</span></div>
      <div class="row"><span class="label">Decision Threshold</span><span class="value">0.500</span></div>
    </div>

    <div class="panel" style="grid-column: span 2;">
      <h3>Defect Details</h3>
      <div class="row"><span class="label">Defect Type</span><span class="value">Anomalous pattern deviation (unsupervised detection &mdash; not classified into a named defect category)</span></div>
      <div class="row"><span class="label">Localization</span><span class="value">See red-circled region and zoomed inset on the image above</span></div>
      <div class="row"><span class="label">Note</span><span class="value">Component-level identification (e.g. "IC", "Resistor") requires a populated-board training dataset; this model was trained on bare-board copper trace imagery</span></div>
    </div>
  </div>

  <div class="recommendation">
    <strong>Recommendation:</strong> {recommendation}
  </div>
</div>
</body>
</html>
"""

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_html_path
