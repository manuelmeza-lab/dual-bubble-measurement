"""
Calibration module — Spatial calibration using a known reference object.

Uses the same two-stage detection pipeline to measure a reference
object (e.g., a 4mm metallic sphere) and compute the pixels-per-millimeter
ratio for subsequent bubble measurements.
"""

from __future__ import annotations

import logging

import numpy as np

from bubble_cv.detection import detect_bubble

logger = logging.getLogger(__name__)


def calibrate(
    frame: np.ndarray,
    known_diameter_mm: float = 4.0,
    min_radius: int = 50,
    max_radius: int = 500,
    **detection_kwargs,
) -> float | None:
    """Compute px/mm ratio from a reference object of known size.

    Detects the reference object in the frame and computes the
    calibration ratio based on its known diameter.

    Args:
        frame: BGR image containing the reference object.
        known_diameter_mm: Known diameter of the reference object (mm).
        min_radius: Minimum expected radius of the reference object (px).
        max_radius: Maximum expected radius of the reference object (px).
        **detection_kwargs: Additional keyword arguments passed to
            detect_bubble() (e.g., hough_param1, clip_limit).

    Returns:
        Pixels-per-millimeter ratio, or None if detection fails.
    """
    detection = detect_bubble(
        frame,
        px_to_mm=None,  # No calibration yet
        min_radius=min_radius,
        max_radius=max_radius,
        **detection_kwargs,
    )

    if detection is None:
        logger.error("Calibration failed: could not detect reference object.")
        return None

    measured_diameter_px = detection.equiv_diameter_px
    if measured_diameter_px <= 0:
        logger.error("Calibration failed: measured diameter is zero.")
        return None

    px_to_mm = measured_diameter_px / known_diameter_mm

    logger.info(
        "Calibration result: %.2f px/mm "
        "(measured %.1f px for %.1f mm reference, method=%s)",
        px_to_mm, measured_diameter_px, known_diameter_mm, detection.method,
    )

    return px_to_mm


def validate_calibration(
    measured_diameter_mm: float,
    known_diameter_mm: float,
) -> dict:
    """Validate calibration accuracy by comparing measured vs. known values.

    Args:
        measured_diameter_mm: Diameter measured using the calibration ratio.
        known_diameter_mm: True known diameter.

    Returns:
        Dictionary with absolute error, relative error percentage,
        and a pass/fail status (pass if error < 5%).
    """
    abs_error = abs(measured_diameter_mm - known_diameter_mm)
    rel_error_pct = (abs_error / known_diameter_mm) * 100.0

    status = "PASS" if rel_error_pct < 5.0 else "FAIL"

    result = {
        "measured_mm": round(measured_diameter_mm, 4),
        "known_mm": known_diameter_mm,
        "absolute_error_mm": round(abs_error, 4),
        "relative_error_pct": round(rel_error_pct, 2),
        "status": status,
    }

    if status == "PASS":
        logger.info(
            "Calibration validation PASSED: error=%.2f%%", rel_error_pct
        )
    else:
        logger.warning(
            "Calibration validation FAILED: error=%.2f%% (threshold=5%%)",
            rel_error_pct,
        )

    return result
