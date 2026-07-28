#!/usr/bin/env python3
"""
analyze_image.py — Process single images or a directory of images.

Detects bubbles using the two-stage Hough+Ellipse pipeline and exports
geometric measurements to CSV.

Usage:
    # Single image with manual calibration (100 px/mm):
    python analyze_image.py --input image.png --calibration 100.0

    # Directory of images with visualization:
    python analyze_image.py --input DinoLite/Default/ --calibration 100.0 --visualize

    # Auto-calibrate using a reference image (4mm sphere):
    python analyze_image.py --input image.png --calibrate-from ref.png --ref-diameter 4.0

    # Process without calibration (pixel units only):
    python analyze_image.py --input image.png
"""

import argparse
import logging
import sys
from pathlib import Path

from bubble_cv.calibration import calibrate
from bubble_cv.detection import detect_bubble
from bubble_cv.io_utils import list_images, load_image, save_csv
from bubble_cv.visualization import save_annotated_frame

logger = logging.getLogger("bubble_cv")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="BubbleCV — Analyze bubble geometry from microscope images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input/Output
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to a single image or a directory of images.",
    )
    parser.add_argument(
        "--output", "-o", default="results_images.csv",
        help="Output CSV file path (default: results_images.csv).",
    )

    # Calibration
    cal_group = parser.add_mutually_exclusive_group()
    cal_group.add_argument(
        "--calibration", "-c", type=float, default=None,
        help="Manual calibration ratio in px/mm.",
    )
    cal_group.add_argument(
        "--calibrate-from", type=str, default=None,
        help="Path to a reference image for auto-calibration.",
    )
    parser.add_argument(
        "--ref-diameter", type=float, default=4.0,
        help="Known diameter of reference object in mm (default: 4.0).",
    )

    # Detection parameters
    parser.add_argument(
        "--min-radius", type=int, default=50,
        help="Minimum bubble radius in pixels (default: 50).",
    )
    parser.add_argument(
        "--max-radius", type=int, default=500,
        help="Maximum bubble radius in pixels (default: 500).",
    )
    parser.add_argument(
        "--clip-limit", type=float, default=3.0,
        help="CLAHE clip limit for contrast enhancement (default: 3.0).",
    )

    # Visualization
    parser.add_argument(
        "--visualize", "-v", action="store_true",
        help="Save annotated images with detection overlay.",
    )
    parser.add_argument(
        "--vis-dir", type=str, default="annotated",
        help="Directory for annotated images (default: annotated/).",
    )

    # Logging
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging output.",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for image analysis."""
    args = parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Determine calibration
    px_to_mm = args.calibration

    if args.calibrate_from:
        logger.info("Auto-calibrating from: %s", args.calibrate_from)
        ref_frame = load_image(args.calibrate_from)
        if ref_frame is None:
            logger.error("Failed to load calibration image.")
            return 1
        px_to_mm = calibrate(
            ref_frame,
            known_diameter_mm=args.ref_diameter,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
            clip_limit=args.clip_limit,
        )
        if px_to_mm is None:
            logger.error("Auto-calibration failed.")
            return 1
        logger.info("Calibration: %.2f px/mm", px_to_mm)

    if px_to_mm is None:
        logger.warning(
            "No calibration provided. Results will be in pixel units only."
        )

    # Collect input images
    input_path = Path(args.input)
    if input_path.is_dir():
        image_paths = list_images(input_path)
    elif input_path.is_file():
        image_paths = [input_path]
    else:
        logger.error("Input path does not exist: %s", input_path)
        return 1

    if not image_paths:
        logger.error("No images found to process.")
        return 1

    # Process each image
    results = []
    processed = 0
    failed = 0

    for img_path in image_paths:
        frame = load_image(img_path)
        if frame is None:
            failed += 1
            continue

        detection = detect_bubble(
            frame,
            px_to_mm=px_to_mm,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
            clip_limit=args.clip_limit,
        )

        if detection is None:
            logger.warning("No bubble detected in: %s", img_path.name)
            failed += 1
            continue

        # Build result row (filename first for easy identification)
        row = {"filename": img_path.name, **detection.to_dict()}
        results.append(row)
        processed += 1

        logger.info(
            "  %s: D=%.1f px, ecc=%.3f, method=%s",
            img_path.name,
            detection.equiv_diameter_px,
            detection.eccentricity,
            detection.method,
        )

        # Save annotated image if requested
        if args.visualize:
            vis_path = Path(args.vis_dir) / f"annotated_{img_path.name}"
            save_annotated_frame(frame, detection, vis_path)

    # Export results
    if results:
        save_csv(results, args.output)
        logger.info(
            "Done. Processed: %d, Failed: %d, Output: %s",
            processed, failed, args.output,
        )
    else:
        logger.error("No results to export.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
