"""
grid_segmentation.py
---------------------
Purpose: Split a full PCB image into overlapping grid cells so that small
components (resistors, ICs, etc.) get proportionally more pixel resolution
during YOLO detection, instead of being compressed into a handful of pixels
in a single full-board image.

Also tracks each grid cell's position in the original image, so detections
can later be mapped back to full-PCB coordinates for heatmaps and
root-cause tracking (per-grid defect history).
"""

import cv2
import numpy as np


def split_into_grids(image, rows, cols, overlap_ratio=0.1):
    """
    Splits an image into a rows x cols grid of overlapping cells.

    Why overlap matters:
    Without it, a component sitting exactly on a grid boundary line gets
    sliced in half across two cells, and YOLO may fail to detect either
    half properly. Overlap ensures boundary-straddling components have
    a higher chance of appearing whole in at least one cell.

    Args:
        image (np.ndarray): full preprocessed PCB image, shape (H, W, C)
        rows (int): number of grid rows (from config.yaml)
        cols (int): number of grid columns (from config.yaml)
        overlap_ratio (float): fraction of cell size to overlap with neighbors

    Returns:
        list of dict: each dict contains:
            'grid_id' (int): sequential grid number (0-indexed, left-to-right, top-to-bottom)
            'image' (np.ndarray): the cropped grid cell image
            'x_offset' (int): x-coordinate of this cell's top-left corner in the ORIGINAL image
            'y_offset' (int): y-coordinate of this cell's top-left corner in the ORIGINAL image
    """
    h, w = image.shape[:2]

    # Base cell size before overlap is applied
    cell_h = h // rows
    cell_w = w // cols

    # Overlap in pixels (added to each cell's height/width)
    overlap_h = int(cell_h * overlap_ratio)
    overlap_w = int(cell_w * overlap_ratio)

    grids = []
    grid_id = 0

    for row in range(rows):
        for col in range(cols):
            # Base (non-overlapping) boundaries for this cell
            y_start = row * cell_h
            y_end = y_start + cell_h
            x_start = col * cell_w
            x_end = x_start + cell_w

            # Expand boundaries by overlap amount, clamped to image edges
            y_start_overlap = max(0, y_start - overlap_h)
            y_end_overlap = min(h, y_end + overlap_h)
            x_start_overlap = max(0, x_start - overlap_w)
            x_end_overlap = min(w, x_end + overlap_w)

            cell_image = image[y_start_overlap:y_end_overlap, x_start_overlap:x_end_overlap]

            grids.append({
                "grid_id": grid_id,
                "image": cell_image,
                "x_offset": x_start_overlap,
                "y_offset": y_start_overlap
            })
            grid_id += 1

    return grids

def local_to_global_coords(bbox, x_offset, y_offset):
    """
    Converts a bounding box from grid-cell-local coordinates to
    full-PCB-image global coordinates.

    Why this matters:
    YOLO detects components within a single cropped grid cell, so its
    output bounding box is relative to that small cell (e.g., x=20 means
    "20 pixels from this cell's left edge", NOT the full board's left edge).
    Everything downstream -- heatmap overlay on the full PCB image, database
    records, root cause tracking by physical location -- needs coordinates
    relative to the FULL board. This function performs that translation
    using the offset recorded when the cell was created in split_into_grids().

    Args:
        bbox (tuple): (x1, y1, x2, y2) in LOCAL grid-cell coordinates
        x_offset (int): this cell's x-offset in the original image
        y_offset (int): this cell's y-offset in the original image

    Returns:
        tuple: (x1, y1, x2, y2) in GLOBAL full-image coordinates
    """
    x1, y1, x2, y2 = bbox

    global_x1 = x1 + x_offset
    global_y1 = y1 + y_offset
    global_x2 = x2 + x_offset
    global_y2 = y2 + y_offset

    return (global_x1, global_y1, global_x2, global_y2)

def draw_grid_overlay(image, grids):
    """
    Draws grid cell boundaries and ID labels on a copy of the full image.
    Used for debugging, dashboard "Grid Layout" view, and interview demos.

    Args:
        image (np.ndarray): the full original image (NOT preprocessed/normalized --
            use the raw BGR image here since this is for human viewing, not model input)
        grids (list of dict): output from split_into_grids()

    Returns:
        np.ndarray: a copy of the image with grid rectangles and ID labels drawn on it
    """
    overlay = image.copy()

    for cell in grids:
        x1 = cell["x_offset"]
        y1 = cell["y_offset"]
        h, w = cell["image"].shape[:2]
        x2 = x1 + w
        y2 = y1 + h

        # Draw rectangle boundary for this grid cell
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # Label the grid cell with its ID number
        label_position = (x1 + 5, y1 + 20)
        cv2.putText(
            overlay,
            f"Grid {cell['grid_id']}",
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 255, 0),
            thickness=1
        )

    return overlay