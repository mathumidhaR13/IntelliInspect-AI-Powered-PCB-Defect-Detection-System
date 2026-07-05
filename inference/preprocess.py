"""
preprocess.py
--------------
Purpose: Standardize raw PCB images before they enter the AI pipeline.
Operations: Resize -> (optional Grayscale) -> Denoise -> Align -> Normalize

Used in TWO contexts:
1. Batch mode -> preprocessing all 90 training images before PatchCore training
2. Single-image mode -> preprocessing one uploaded image at inference time
"""

import cv2
import numpy as np
import yaml
import os


def load_config(config_path="config.yaml"):
    """
    Loads pipeline parameters from config.yaml.
    Why: Keeps image_size, thresholds, and paths configurable without touching code.
    Interview point: shows separation of configuration from logic (12-factor app principle).
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def resize_image(image, target_size):
    """
    Resizes an image to target_size (width, height).

    Why smart interpolation matters:
    - Downscaling (image larger than target) uses INTER_AREA -> preserves detail, reduces noise/aliasing.
    - Upscaling (image smaller than target) uses INTER_LINEAR -> smoother enlargement.
    Using the wrong method can introduce blur artifacts that PatchCore may
    mistake for texture anomalies, causing false positives.

    Args:
        image (np.ndarray): input image (BGR, as read by OpenCV)
        target_size (tuple): (width, height) from config.yaml

    Returns:
        np.ndarray: resized image
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Decide interpolation method based on whether we are shrinking or enlarging
    if target_w < w or target_h < h:
        interpolation = cv2.INTER_AREA      # shrinking
    else:
        interpolation = cv2.INTER_LINEAR    # enlarging

    resized = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
    return resized

def convert_grayscale(image):
    """
    Converts a BGR image to grayscale.

    Why this is OPTIONAL and off by default (config: use_grayscale=false):
    PatchCore's backbone (WideResNet, pretrained on ImageNet) expects 3-channel
    RGB input. Converting to grayscale here is only for debugging/visualization
    or for future PCB variants where color is not diagnostically useful.
    The actual PatchCore training/inference pipeline will use the RGB image.

    Args:
        image (np.ndarray): input BGR image

    Returns:
        np.ndarray: single-channel grayscale image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def denoise_image(image, strength):
    """
    Removes sensor noise while preserving fine edges/textures using
    Non-Local Means Denoising (cv2.fastNlMeansDenoisingColored).

    Why NOT a simple Gaussian/median blur:
    A basic blur averages nearby pixels blindly, which also smudges real
    PCB trace edges and defect boundaries. Non-Local Means instead searches
    the whole image for similar patches and averages only genuinely similar
    regions -- removing random noise while keeping fine repetitive textures
    (like PCB copper traces) sharp. This is critical because PatchCore is
    sensitive to exactly this kind of fine texture detail.

    Args:
        image (np.ndarray): input BGR image
        strength (int): denoising strength from config.yaml (higher = more smoothing)

    Returns:
        np.ndarray: denoised image
    """
    denoised = cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h=strength,          # filter strength for luminance
        hColor=strength,     # filter strength for color channels
        templateWindowSize=7,
        searchWindowSize=21
    )
    return denoised

def align_image(image, reference_image, max_features=500, good_match_percent=0.15):
    """
    Aligns 'image' to 'reference_image' using ORB feature matching + homography.

    Why this matters:
    A real PCB placed on an inspection fixture is never pixel-perfect aligned --
    small rotation/shift tolerance always exists. If left uncorrected, grid
    segmentation (Step 3) and YOLO component boxes would land on inconsistent
    physical locations across different boards, breaking comparability.

    Why ORB over SIFT/SURF:
    ORB is free (no patent licensing concerns), fast, and rotation-invariant --
    important since this logic may eventually run on an edge device (Vyqor)
    where compute and latency budgets are tight.

    Args:
        image (np.ndarray): the new image to align (e.g. uploaded PCB)
        reference_image (np.ndarray): the fixed reference/template good image
        max_features (int): max ORB keypoints to detect
        good_match_percent (float): fraction of top matches to keep as "good"

    Returns:
        np.ndarray: image warped to align with reference_image
    """
    # Convert both images to grayscale for feature detection (ORB works on intensity, not color)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    # Step 1: Detect ORB keypoints and descriptors in both images
    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(img_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(ref_gray, None)

    # Safety check: if too few features found, skip alignment and return original
    if descriptors1 is None or descriptors2 is None:
        return image

    # Step 2: Match features between the two images using Hamming distance
    # Step 2: Match features between the two images using Hamming distance
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = list(matcher.match(descriptors1, descriptors2))

    # Step 3: Keep only the best matches (lowest distance = most reliable)
    matches.sort(key=lambda x: x.distance)
    num_good_matches = int(len(matches) * good_match_percent)
    matches = matches[:num_good_matches]

    # Not enough reliable matches -> alignment would be unstable, return original
    if len(matches) < 4:
        return image

    # Step 4: Extract matched point coordinates
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Step 5: Compute homography (the transform matrix) using RANSAC
    # RANSAC discards outlier matches that don't fit the dominant transform
    h_matrix, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Safety check: if homography computation failed
    if h_matrix is None:
        return image

    # Step 6: Warp the image using the computed homography
    height, width = reference_image.shape[:2]
    aligned = cv2.warpPerspective(image, h_matrix, (width, height))

    return aligned

def normalize_image(image, mean, std):
    """
    Converts an OpenCV BGR image into a PyTorch-ready normalized tensor array.

    Steps (order matters):
    1. BGR -> RGB: OpenCV loads images as BGR by default; PyTorch/ImageNet
       pretrained models expect RGB channel order.
    2. Scale 0-255 -> 0.0-1.0: neural networks train more stably on small
       float ranges rather than raw 0-255 integers.
    3. Standardize using ImageNet mean/std: centers the data to match the
       exact distribution the pretrained WideResNet backbone was trained on.
       Skipping this step is a common silent bug -- the model still runs,
       but produces meaningless feature embeddings.
    4. Reorder (H, W, C) -> (C, H, W): PyTorch expects channel-first format,
       while OpenCV/numpy images are channel-last by default.

    Args:
        image (np.ndarray): input BGR image, values 0-255
        mean (list): per-channel mean, from config.yaml (ImageNet values)
        std (list): per-channel std, from config.yaml (ImageNet values)

    Returns:
        np.ndarray: normalized image, shape (C, H, W), dtype float32,
                     ready to be converted into a torch.Tensor
    """
    # Step 1: BGR -> RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 2: Scale pixel values from [0, 255] to [0.0, 1.0]
    image_float = image_rgb.astype(np.float32) / 255.0

    # Step 3: Standardize using ImageNet mean/std (per channel)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    image_normalized = (image_float - mean) / std

    # Step 4: Reorder from (Height, Width, Channels) to (Channels, Height, Width)
    image_chw = np.transpose(image_normalized, (2, 0, 1))

    return image_chw

def preprocess_pipeline(image, config, reference_image=None):
    """
    Runs the full preprocessing pipeline in the correct order:
    Resize -> Denoise -> Align (optional) -> Normalize

    This is the SINGLE function every other module (training, inference,
    dashboard) should call. They should never need to know the internal
    step order -- that's an implementation detail hidden here. If the
    order ever changes, it only needs to be updated in this one place.

    Args:
        image (np.ndarray): raw input image (BGR, as read by cv2.imread)
        config (dict): loaded config.yaml dictionary (from load_config())
        reference_image (np.ndarray, optional): reference/template good image
            used for alignment. Required only if alignment_enabled=true in config.
            Not needed for training-set preprocessing (Step 6), since we don't
            align good images to each other -- only new inference images get
            aligned against a fixed reference.

    Returns:
        np.ndarray: fully preprocessed image, shape (C, H, W), ready for
                     conversion to a torch.Tensor and model input.
    """
    cfg = config["preprocessing"]

    # Step 1: Resize to standard model input size
    image = resize_image(image, tuple(cfg["image_size"]))

    # Step 2: Denoise (removes sensor noise, preserves real edges/texture)
    image = denoise_image(image, cfg["denoise_strength"])

    # Step 3: Align (optional -- only if enabled AND a reference image is provided)
    if cfg["alignment_enabled"] and reference_image is not None:
        reference_image = resize_image(reference_image, tuple(cfg["image_size"]))
        image = align_image(image, reference_image)

    # Step 4: Normalize -> PyTorch-ready (C, H, W) float32 array
    image = normalize_image(image, cfg["normalize_mean"], cfg["normalize_std"])

    return image