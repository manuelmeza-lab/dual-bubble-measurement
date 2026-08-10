"""
Calibration module — Spatial calibration using a known reference object.

Uses a dedicated single-object detector to measure a reference object
(e.g., a 4 mm metallic sphere) and compute the pixels-per-millimetre
ratio for subsequent bubble measurements.

The calibration detector is completely independent of the dual pendant-drop
pipeline: it works on the full frame, uses caller-supplied Hough radius
limits, and applies no drop-specific geometry filters.
"""

from __future__ import annotations

import logging

import numpy as np

from bubble_cv.detection import detect_reference_object

logger = logging.getLogger(__name__)


def calibrate(
    frame: np.ndarray,
    known_diameter_mm: float = 4.0,
    min_radius: int = 50,
    max_radius: int = 500,
    **detection_kwargs,
) -> float | None:
    """Compute px/mm ratio from a reference object of known size.

    Detects the reference object in the frame using
    :func:`~bubble_cv.detection.detect_reference_object` and computes the
    calibration ratio based on its known diameter.

    Args:
        frame             : BGR image containing the reference object.
        known_diameter_mm : Known diameter of the reference object (mm).
        min_radius        : Minimum expected radius of the reference object (px).
        max_radius        : Maximum expected radius of the reference object (px).
        **detection_kwargs: Additional keyword arguments forwarded to
                            ``detect_reference_object()``
                            (e.g. ``blur_kernel``, ``clip_limit``,
                            ``hough_param1``, ``hough_param2``).

    Returns:
        Pixels-per-millimetre ratio, or ``None`` if detection fails.
    """
    measured_diameter_px = detect_reference_object(
        frame,
        min_radius=min_radius,
        max_radius=max_radius,
        **detection_kwargs,
    )

    if measured_diameter_px is None:
        logger.error("Calibration failed: could not detect reference object.")
        return None

    if measured_diameter_px <= 0:
        logger.error(
            "Calibration failed: measured diameter is %.4f px (must be > 0).",
            measured_diameter_px,
        )
        return None

    px_to_mm = measured_diameter_px / known_diameter_mm

    logger.info(
        "Calibration: reference diameter detected = %.2f px  |  "
        "known diameter = %.2f mm  |  calibration = %.4f px/mm",
        measured_diameter_px, known_diameter_mm, px_to_mm,
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
