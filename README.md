# IntelliInspect

**AI-powered visual inspection system for PCB (Printed Circuit Board) manufacturing**, using unsupervised anomaly detection to catch defects, localize them on the board, and flag repeated failure patterns for root-cause investigation.

---

## Overview

IntelliInspect analyzes PCB images and automatically determines whether a board is **PASS** or **FAIL**, without ever being shown a single defective image during training. It learns what a "normal" board looks like, then flags anything that deviates from that — the same core idea used in real industrial visual-inspection systems.

Given an input image, the system produces:
- An **anomaly score** (0–1) indicating how different the board is from normal
- A **pixel-level heatmap** showing *where* the difference is
- A **binary defect mask** outlining the exact defect region
- A **PASS/FAIL label**, based on a configurable threshold
- A **marked-up output image** with the defect circled and the score/label overlaid
- Automatic **repeat-defect alerting**, so if the same failure pattern shows up on consecutive boards, the system flags it as a possible root cause

---

## How It Works

1. **Training** — A [PatchCore](https://arxiv.org/abs/2106.08265) model is trained *only* on images of known-good PCBs. It builds a "memory bank" of normal-patch features — this is the entire knowledge base the model has about what a good board looks like.
2. **Inference** — A new image is compared patch-by-patch against this memory bank. Regions that don't match anything in the memory bank get a high anomaly score.
3. **Grid Segmentation** — The board is optionally split into grid cells to help localize *which section* of the board is problematic.
4. **Thresholding** — The raw anomaly score is compared against a threshold (configurable in `config.yaml`) to make the final PASS/FAIL call — explainable and tunable, rather than relying purely on an internal auto-threshold.
5. **Visualization** — The anomaly heatmap is blended onto the original image, the defect region is outlined with a contour, and the score/label are stamped on the image.
6. **Root Cause Monitoring** — Every inspection is logged to a SQLite database. If the last 3 inspections all failed with defects clustered in the same location, the system automatically triggers an alert.

---

## Project Structure

```
IntelliInspect/
├── inference/
│   ├── anomaly_inference.py     # Load model, run inference, visualize results
│   ├── preprocess.py             # Config loading, image resizing/normalization
│   ├── grid_segmentation.py      # Splits PCB image into grid cells
│   └── report_generator.py       # Generates inspection reports
├── models/                       # Model architecture / wrapper code
├── root_cause/
│   └── root_cause_engine.py      # Detects repeated defect patterns, triggers alerts
├── training/
│   ├── train_patchcore.py        # Trains the PatchCore anomaly detection model
│   ├── train_yolo.py             # Trains YOLO (defect/component detection)
│   └── config_training.yaml
├── tests/
│   ├── detect_defect.py          # End-to-end single-image inspection test
│   ├── test_anomaly_inference.py
│   ├── test_grid_segmentation.py
│   ├── test_patchcore.py
│   ├── test_pipeline.py
│   ├── test_preprocessing.py
│   └── test_yolo.py
├── utils/                        # Shared helper functions
├── config.yaml                   # Central configuration (thresholds, paths, etc.)
└── requirements.txt
```

> **Note:** `venv/`, `weights/`, `results/`, and dataset folders (e.g. `pcb_anomaly/dataset/`) are intentionally excluded from version control — see [Setup](#setup) below.

---

## Tech Stack

- **Python 3.13**
- **[Anomalib](https://github.com/openvinotoolkit/anomalib)** — PatchCore anomaly detection model
- **YOLOv8** — component/defect detection
- **OpenCV** — image processing, heatmap overlay, contour drawing
- **NumPy**
- **SQLite** — inspection history logging
- **PyTorch** — underlying deep learning framework

---

## Setup

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd IntelliInspect
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your dataset
This repo does **not** include the raw PCB image dataset. Create the following structure and populate it with your own images:
```
pcb_anomaly/dataset/
├── raw_good/       # Known-good PCB images (used for training)
├── test_defect/    # Defective PCB images (used for testing/demo)
└── ...
```

### 5. Add model weights
Pretrained/trained weights (e.g. `yolov8n.pt`, PatchCore checkpoints) are not included due to file size. Either:
- Train them yourself using the scripts in `training/`, or
- Download them from `<link to your hosted weights, e.g. Google Drive / HuggingFace>` and place them in `weights/`

---

## Usage

**Run inspection on a single image:**
```bash
python tests/detect_defect.py "pcb_anomaly/dataset/test_defect/defect_001.jpg"
```

This will:
- Load the trained model
- Score the image and produce a PASS/FAIL result
- Save a marked-up visualization with the defect highlighted
- Log the result to the inspection history database
- Trigger an alert if the same defect pattern has occurred on the last 3 boards

**Train the PatchCore model:**
```bash
python training/train_patchcore.py
```

---

## Alerting / Root Cause Monitoring

Every inspection is recorded (board ID, score, PASS/FAIL, severity, defect location, timestamp) in `database/inspection_history.db`. After each inspection, the system checks whether the last 3 inspections all failed with defects clustered in the same location. If so, it:
- Prints a console alert banner
- Writes a permanent record to `alerts_log.txt`

> In its current form, alerts are logged locally rather than delivered via email/SMS. The full trigger and decision logic is implemented; only the final delivery channel (e.g. SMTP or an SMS API like Twilio) would need to be added for production use.

---

## Roadmap / Possible Extensions
- Wire up real email/SMS delivery for alerts
- Web-based dashboard for uploading images and viewing inspection history
- Expand grid segmentation into a full defect-location classification system
- Model retraining pipeline as new "good" images are collected

---

## License
Specify your license here (e.g. MIT).
