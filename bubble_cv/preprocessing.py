"""
Preprocessing module — Contrast enhancement and channel selection.

Designed for low-illumination DinoLite microscope images where the
bubble boundary has very low contrast against the dark background.

Key technique: CLAHE (Contrast Limited Adaptive Histogram Equalization)
operates on local tiles rather than the whole image, making it far more
effective than global histogram equalization for images with narrow
dynamic range.
"""

import cv2
import numpy as np


def enhance_contrast(
    gray: np.ndarray,
    clip_limit: float = 3.0,
    tile_size: int = 8,
) -> np.ndarray:
    """Apply CLAHE to boost local contrast.

    Essential preprocessing step for low-illumination DinoLite images
    where the bubble and background have similar intensity values.

    Args:
        gray: Grayscale input image (uint8).
        clip_limit: Contrast limit for CLAHE. Higher values allow
            more contrast enhancement but may amplify noise.
        tile_size: Size of the grid tiles for local histogram
            equalization. Smaller tiles = more local adaptation.

    Returns:
        Contrast-enhanced grayscale image (uint8).
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size),
    )
    return clahe.apply(gray)


def select_best_channel(frame: np.ndarray) -> tuple[np.ndarray, str]:
    """Automatically select the color channel with highest contrast.

    Evaluates each BGR channel by standard deviation and returns the
    one with the most variation (best chance of separating bubble
    from background).

    Args:
        frame: BGR color image (uint8, 3-channel).

    Returns:
        Tuple of (selected channel as grayscale image, channel name).
    """
    b, g, r = cv2.split(frame)
    contrasts = {
        "blue": np.std(b),
        "green": np.std(g),
        "red": np.std(r),
    }
    best_name = max(contrasts, key=contrasts.get)
    channels = {"blue": b, "green": g, "red": r}
    return channels[best_name], best_name


def preprocess(
    frame: np.ndarray,
    blur_kernel: int = 7,
    clip_limit: float = 3.0,
    tile_size: int = 8,
) -> np.ndarray:
    """Full preprocessing pipeline for bubble detection.

    Steps:
        1. Auto-select the best color channel
        2. Apply CLAHE contrast enhancement
        3. Apply Gaussian blur to reduce noise

    Args:
        frame: BGR color image (uint8, 3-channel).
        blur_kernel: Size of the Gaussian blur kernel (must be odd).
        clip_limit: CLAHE contrast limit.
        tile_size: CLAHE tile grid size.

    Returns:
        Preprocessed grayscale image ready for detection.
    """
    # Step 1: Select best channel
    gray, _channel_name = select_best_channel(frame)

    # Step 2: Enhance contrast with CLAHE
    enhanced = enhance_contrast(gray, clip_limit, tile_size)

    # Step 3: Gaussian blur to smooth noise while preserving edges
    blurred = cv2.GaussianBlur(enhanced, (blur_kernel, blur_kernel), 0)

    return blurred
