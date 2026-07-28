"""
I/O utilities module — CSV export and video frame iteration.

Handles reading images and video frames, and exporting detection
results to CSV format for downstream analysis.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import pandas as pd

from bubble_cv.detection import BubbleDetection

logger = logging.getLogger(__name__)

# Supported image file extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

# Supported video file extensions
VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv"}


def load_image(path: str | Path) -> np.ndarray | None:
    """Load a single image from disk.

    Args:
        path: Path to the image file.

    Returns:
        BGR image as numpy array, or None if loading fails.
    """
    path = Path(path)
    if not path.exists():
        logger.error("Image file not found: %s", path)
        return None

    frame = cv2.imread(str(path))
    if frame is None:
        logger.error("Failed to load image (may be corrupt): %s", path)
        return None

    logger.debug("Loaded image: %s (%dx%d)", path.name, frame.shape[1], frame.shape[0])
    return frame


def list_images(directory: str | Path) -> list[Path]:
    """List all image files in a directory, sorted by name.

    Args:
        directory: Path to the directory containing images.

    Returns:
        Sorted list of image file paths.
    """
    directory = Path(directory)
    if not directory.is_dir():
        logger.error("Not a directory: %s", directory)
        return []

    images = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    logger.info("Found %d images in %s", len(images), directory)
    return images


def frame_iterator(
    video_path: str | Path,
    skip: int = 1,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Iterate over frames of a video file.

    Args:
        video_path: Path to the video file (.mov, .mp4, etc.).
        skip: Process every Nth frame. Use skip=1 for all frames,
            skip=30 to process one per second at 30fps, etc.

    Yields:
        Tuples of (frame_number, BGR frame array).
    """
    video_path = Path(video_path)
    if not video_path.exists():
        logger.error("Video file not found: %s", video_path)
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Failed to open video: %s", video_path)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(
        "Processing video: %s (%d frames, skip=%d)",
        video_path.name, total_frames, skip,
    )

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % skip == 0:
            yield frame_num, frame

        frame_num += 1

    cap.release()
    logger.info("Finished processing video: %d frames read.", frame_num)


def save_csv(
    results: list[dict],
    output_path: str | Path,
    include_header: bool = True,
) -> None:
    """Export detection results to a CSV file.

    Args:
        results: List of dictionaries (from BubbleDetection.to_dict()).
        output_path: Path for the output CSV file.
        include_header: Whether to write column headers.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        logger.warning("No results to save.")
        return

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d results to %s", len(results), output_path)


def append_csv_row(
    result: dict,
    output_path: str | Path,
    write_header: bool = False,
) -> None:
    """Append a single detection result row to a CSV file.

    Useful for streaming results during video processing.

    Args:
        result: Dictionary from BubbleDetection.to_dict().
        output_path: Path to the CSV file.
        write_header: Whether to write column headers (set True for first row).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if output_path.exists() and not write_header else "w"
    with open(output_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if write_header or not output_path.exists():
            writer.writeheader()
        writer.writerow(result)
