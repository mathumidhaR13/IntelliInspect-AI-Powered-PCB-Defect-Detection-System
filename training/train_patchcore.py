"""
train_patchcore.py
--------------------
Purpose: Train a PatchCore anomaly detection model using ONLY good PCB images.

Key concept: PatchCore is NOT trained with gradient descent like a normal
neural network. Instead, it:
1. Runs a pretrained CNN backbone (WideResNet) over every good training image
   to extract local feature "patches" from mid-level layers.
2. Collects all these normal feature patches into a "memory bank".
3. Uses coreset subsampling to keep only the most representative patches
   (for speed), discarding redundant near-duplicates.

At inference time, a new image's patches are compared against this memory
bank using nearest-neighbor distance -- patches far from anything in the
memory bank are flagged as anomalous. This is why PatchCore needs no
defect images to train: it only needs to know what "normal" looks like.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

from inference.preprocess import load_config


def train_patchcore():
    # Load all parameters from config.yaml -- no hardcoded values
    config = load_config(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    pc_cfg = config["patchcore"]
    paths_cfg = config["paths"]

    project_root = os.path.join(os.path.dirname(__file__), "..")
    dataset_root = os.path.join(project_root, "dataset")

    print("Setting up datamodule from dataset/raw_good and dataset/test_defect ...")

    # Folder datamodule maps directly onto our existing folder structure.
    # normal_dir = good images (training set)
    # abnormal_dir = defect images (used for validation/test evaluation only,
    #                NEVER used to train the memory bank itself)
    datamodule = Folder(
        name="IntelliInspectPCB",
        root=dataset_root,
        normal_dir=paths_cfg["good_dir_name"],      # "raw_good"
        abnormal_dir=paths_cfg["defect_dir_name"],   # "test_defect"
    )

    print("Initializing PatchCore model...")
    model = Patchcore(
        backbone=pc_cfg["backbone"],
        layers=pc_cfg["layers"],
        coreset_sampling_ratio=pc_cfg["coreset_sampling_ratio"],
        num_neighbors=pc_cfg["num_neighbors"],
    )

    print("Starting training (building the normal-patch memory bank)...")
    engine = Engine(
        max_epochs=1,   # PatchCore only needs ONE pass over the data --
                        # it's collecting features, not doing gradient-based learning.
                        # Training for more epochs would just re-extract the same
                        # features redundantly.
    )
    engine.fit(model=model, datamodule=datamodule)

    print("Training complete. Running evaluation on test_defect images...")
    test_results = engine.test(model=model, datamodule=datamodule)
    print("Test results:", test_results)

    print("PatchCore model and memory bank are ready for inference.")
    return model, engine


if __name__ == "__main__":
    train_patchcore()