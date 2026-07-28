"""
Detection module — Dual pendant-drop detection with spatial classification.

Pipeline overview:
    1. Preprocessing   : Grayscale conversion + CLAHE contrast enhancement +
                         Gaussian blur.
    2. Binary masking  : Global Otsu thresholding → inverted binary mask.
    3. Capillary isolation : Aggressive MORPH_OPEN (elliptical 7×7 kernel,
                             2 iterations) to visually break the metallic
                             needle from each pendant drop.
     4. Contour extraction  : Filter contours by plausible axis size, area,
                             and vertical position; prefer one per half-frame.
    5. Ellipse fitting     : cv2.fitEllipse() on each of the two contours.
    6. Spatial classification:
                             • Left  contour (smaller centroid X) → 'control'
                             • Right contour (larger  centroid X) → 'sample'
    7. Physical conversion : Semi-axes → mm → spheroid geometry if px_to_mm
                             is supplied.

Returns
-------
dict[str, BubbleDetection | None]
    {'control': BubbleDetection | None, 'sample': BubbleDetection | None}

    Returns None for an individual key if that drop could not be detected.
    Returns None (the whole dict) only if fewer than 2 valid contours exist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from bubble_cv.geometry import (
    eccentricity,
    ellipse_area,
    equivalent_diameter,
    spheroid_surface,
    spheroid_volume,
)
from bubble_cv.preprocessing import preprocess

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Morphological constants for capillary isolation
# ---------------------------------------------------------------------------
_CAPILLARY_KERNEL_SIZE: int = 7      # Side length of the elliptical SE
_CAPILLARY_OPEN_ITERS: int = 2       # Number of erosion+dilation cycles


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class BubbleDetection:
    """Result of a pendant-drop detection on a single frame.

    All *_px  fields are in pixel units.
    All *_mm  fields are in millimetres (populated only when px_to_mm is given).
    """

    # ---- Pixel-space measurements -----------------------------------------
    center_x_px: float = 0.0
    center_y_px: float = 0.0
    major_axis_px: float = 0.0
    minor_axis_px: float = 0.0
    angle_deg: float = 0.0
    equiv_diameter_px: float = 0.0
    eccentricity: float = 0.0

    # ---- Physical measurements (mm) — None if uncalibrated ----------------
    major_axis_mm: float | None = None
    minor_axis_mm: float | None = None
    equiv_diameter_mm: float | None = None
    area_mm2: float | None = None
    surface_mm2: float | None = None
    volume_mm3: float | None = None
    radius_eq_mm: float | None = None
    radius_eq_mm2: float | None = None

    # ---- Detection metadata ------------------------------------------------
    confidence: float = 0.0
    method: str = ""
    contour: np.ndarray = field(default_factory=lambda: np.array([]))

    # ---- Tracking quality control -----------------------------------------
    tracking_valid: bool = True
    rejection_reason: str = ""

    # ---- Spatial label (set by detect_bubbles) ----------------------------
    label: str = ""          # 'control' or 'sample'

    def to_dict(self) -> dict:
        """Convert to a flat dictionary suitable for CSV export."""
        return {
            "label": self.label,
            "center_x_px": round(self.center_x_px, 2),
            "center_y_px": round(self.center_y_px, 2),
            "major_axis_px": round(self.major_axis_px, 2),
            "minor_axis_px": round(self.minor_axis_px, 2),
            "angle_deg": round(self.angle_deg, 2),
            "equiv_diameter_px": round(self.equiv_diameter_px, 2),
            "eccentricity": round(self.eccentricity, 4),
            "major_axis_mm": round(self.major_axis_mm, 4) if self.major_axis_mm is not None else None,
            "minor_axis_mm": round(self.minor_axis_mm, 4) if self.minor_axis_mm is not None else None,
            "equiv_diameter_mm": round(self.equiv_diameter_mm, 4) if self.equiv_diameter_mm is not None else None,
            "area_mm2": round(self.area_mm2, 4) if self.area_mm2 is not None else None,
            "surface_mm2": round(self.surface_mm2, 4) if self.surface_mm2 is not None else None,
            "volume_mm3": round(self.volume_mm3, 4) if self.volume_mm3 is not None else None,
            "radius_eq_mm": round(self.radius_eq_mm, 4) if self.radius_eq_mm is not None else None,
            "radius_eq_mm2": round(self.radius_eq_mm2, 4) if self.radius_eq_mm2 is not None else None,
            "tracking_valid": self.tracking_valid,
            "rejection_reason": self.rejection_reason,
            "confidence": round(self.confidence, 4),
            "method": self.method,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _binary_mask(gray: np.ndarray) -> np.ndarray:
    """Convert a preprocessed grayscale image to a binary mask.

    Uses Otsu's method (global thresholding) with THRESH_BINARY_INV so that
    dark pendant drops on a bright background become white foreground regions.

    Args:
        gray: Preprocessed grayscale image (uint8).

    Returns:
        Binary mask (uint8, values 0 or 255).
    """
    _, mask = cv2.threshold(
        gray,
        0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    return mask


def _isolate_drops(mask: np.ndarray) -> np.ndarray:
    """Break the needle–drop connection via aggressive morphological opening.

    An elliptical structuring element of size ``_CAPILLARY_KERNEL_SIZE`` ×
    ``_CAPILLARY_KERNEL_SIZE`` is used for ``_CAPILLARY_OPEN_ITERS``
    iterations.  The elliptical SE is preferred over a square one because
    it better preserves the round drop body while eroding thin capillary
    bridges.

    Args:
        mask: Binary foreground mask (uint8).

    Returns:
        Morphologically cleaned binary mask (uint8).
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (_CAPILLARY_KERNEL_SIZE, _CAPILLARY_KERNEL_SIZE),
    )
    opened = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=_CAPILLARY_OPEN_ITERS,
    )
    logger.debug(
        "Capillary isolation applied: MORPH_OPEN %dx%d ellipse, %d iter(s).",
        _CAPILLARY_KERNEL_SIZE, _CAPILLARY_KERNEL_SIZE, _CAPILLARY_OPEN_ITERS,
    )
    return opened


def _extract_two_largest_contours(
    mask: np.ndarray,
) -> list[np.ndarray]:
    """Find external contours and return the two with the greatest area.

    Args:
        mask: Binary foreground mask (uint8).

    Returns:
        List of up to 2 contours (numpy arrays), sorted by area descending.
        Returns an empty list if fewer than 2 contours are found.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) < 2:
        logger.warning(
            "Expected 2 drop contours, found %d. Detection aborted.",
            len(contours),
        )
        return []

    # Sort by area, keep the two largest
    sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)
    top_two = sorted_cnts[:2]
    areas = [cv2.contourArea(c) for c in top_two]
    logger.debug("Two largest contour areas: %.1f px², %.1f px²", *areas)
    return top_two


def _fit_single_ellipse(contour: np.ndarray) -> dict | None:
    """Fit an ellipse to a single contour and extract geometric properties.

    Uses ``cv2.fitEllipse()`` which requires at least 5 points.

    Args:
        contour: A single contour as a numpy array of shape (N, 1, 2).

    Returns:
        Dictionary with keys:
            center_x, center_y, major_axis, minor_axis, angle_deg,
            semi_major, semi_minor, eccentricity, equiv_diameter_px.
        Returns None if the contour has fewer than 5 points.
    """
    if len(contour) < 5:
        logger.debug("Contour has only %d points; cannot fit ellipse.", len(contour))
        return None

    (cx, cy), (axis1, axis2), angle = cv2.fitEllipse(contour)

    # cv2.fitEllipse returns *full* axis lengths (diameters)
    major_axis = max(axis1, axis2)
    minor_axis = min(axis1, axis2)
    semi_major = major_axis / 2.0
    semi_minor = minor_axis / 2.0

    return {
        "center_x": cx,
        "center_y": cy,
        "major_axis": major_axis,
        "minor_axis": minor_axis,
        "angle_deg": angle,
        "semi_major": semi_major,
        "semi_minor": semi_minor,
        "eccentricity": eccentricity(semi_major, semi_minor),
        "equiv_diameter_px": equivalent_diameter(semi_major, semi_minor) * 2.0,
    }


def _build_detection(
    ellipse_props: dict,
    contour: np.ndarray,
    px_to_mm: float | None,
    label: str,
) -> BubbleDetection:
    """Populate a BubbleDetection dataclass from fitted ellipse properties.

    Converts pixel measurements to physical units (mm) when ``px_to_mm`` is
    provided.  Volume is computed assuming a symmetric spheroid of revolution
    (oblate when major > minor, prolate when minor > major).

    Args:
        ellipse_props : Output dict from :func:`_fit_single_ellipse`.
        contour       : Full-image contour array for the drop.
        px_to_mm      : Calibration factor (pixels per millimetre).
                        Pass None to skip physical conversion.
        label         : Spatial label — 'control' or 'sample'.

    Returns:
        Fully populated :class:`BubbleDetection` instance.
    """
    result = BubbleDetection()
    result.label = label
    result.method = "otsu+morphopen+ellipse"
    result.confidence = 1.0
    result.contour = contour

    # Pixel-space properties
    result.center_x_px = ellipse_props["center_x"]
    result.center_y_px = ellipse_props["center_y"]
    result.major_axis_px = ellipse_props["major_axis"]
    result.minor_axis_px = ellipse_props["minor_axis"]
    result.angle_deg = ellipse_props["angle_deg"]
    result.eccentricity = ellipse_props["eccentricity"]
    result.equiv_diameter_px = ellipse_props["equiv_diameter_px"]

    # Physical conversion
    if px_to_mm is not None:
        semi_major_mm = ellipse_props["semi_major"] / px_to_mm
        semi_minor_mm = ellipse_props["semi_minor"] / px_to_mm

        result.major_axis_mm = result.major_axis_px / px_to_mm
        result.minor_axis_mm = result.minor_axis_px / px_to_mm
        result.equiv_diameter_mm = result.equiv_diameter_px / px_to_mm
        result.area_mm2 = ellipse_area(semi_major_mm, semi_minor_mm)
        result.surface_mm2 = spheroid_surface(semi_major_mm, semi_minor_mm)
        # Volume assuming symmetric spheroid of revolution
        result.volume_mm3 = spheroid_volume(semi_major_mm, semi_minor_mm)
        result.radius_eq_mm = result.equiv_diameter_mm / 2.0
        result.radius_eq_mm2 = result.radius_eq_mm ** 2
    else:
        result.radius_eq_mm = None
        result.radius_eq_mm2 = None

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_bubbles(
    frame: np.ndarray,
    px_to_mm: float | None = None,
    blur_kernel: int = 7,
    clip_limit: float = 3.0,
) -> dict[str, BubbleDetection | None] | None:
    """Detect two simultaneous pendant drops and classify them spatially.

    Full pipeline
    -------------
    1. ``preprocess()`` → grayscale + CLAHE + Gaussian blur.
    2. Otsu global thresholding → inverted binary mask.
    3. Aggressive MORPH_OPEN (elliptical 7×7 SE, 2 iterations) to sever the
       metallic capillary from the drop body.
    4. Filter contours by plausible size/area/position; prefer one per half.
    5. Fit an ellipse to each contour; compute axes, eccentricity,
       equivalent diameter, and spheroid volume.
    6. Spatial classification (unbreakable rule):
         • centroid X is smaller  →  label = 'control'
         • centroid X is larger   →  label = 'sample'

    Args:
        frame      : BGR colour image from the microscope (uint8, 3-channel).
        px_to_mm   : Calibration factor in pixels per millimetre.
                     When None, only pixel-space measurements are populated.
        blur_kernel: Gaussian blur kernel size (must be a positive odd integer).
        clip_limit : CLAHE clip limit for contrast enhancement.

    Returns:
        ``{'control': BubbleDetection, 'sample': BubbleDetection}``

        Returns ``None`` if fewer than 2 valid contours are found (i.e. the
        dual-drop setup cannot be confirmed).

    Raises:
        ValueError: If ``frame`` is not a 3-channel BGR image.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"Expected a 3-channel BGR image; got shape {frame.shape}."
        )

    # ------------------------------------------------------------------
    # Step 1 — Preprocessing
    # ------------------------------------------------------------------
    gray = preprocess(frame, blur_kernel=blur_kernel, clip_limit=clip_limit)

    # ------------------------------------------------------------------
    # Step 2 — Binary mask via Otsu thresholding
    # ------------------------------------------------------------------
    mask = _binary_mask(gray)

    # ------------------------------------------------------------------
    # Step 3 — Capillary isolation via aggressive morphological opening
    # ------------------------------------------------------------------
    mask = _isolate_drops(mask)

    # ------------------------------------------------------------------
    # Step 4 — Filter contours by physical plausibility
    # ------------------------------------------------------------------
    # Geometric thresholds (pixel units) — tune here if the microscope
    # magnification or image resolution changes.
    min_major_axis_px: float = 20.0
    max_major_axis_px: float = 180.0
    min_minor_axis_px: float = 10.0
    max_minor_axis_px: float = 180.0
    min_area_px: float = 60.0
    max_area_px: float = 20_000.0
    max_center_y_frac: float = 0.60   # drops must be in the upper 60 % of frame

    h, w = mask.shape[:2]
    y_limit = h * max_center_y_frac

    contours_all, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Collect (ellipse_props, contour, area) tuples that pass all filters
    plausible: list[tuple[dict, np.ndarray, float]] = []
    for cnt in contours_all:
        area = cv2.contourArea(cnt)
        if area < min_area_px or area > max_area_px:
            continue

        props = _fit_single_ellipse(cnt)
        if props is None:
            continue

        major = props["major_axis"]
        minor = props["minor_axis"]
        cx    = props["center_x"]
        cy    = props["center_y"]

        if major < min_major_axis_px or major > max_major_axis_px:
            continue
        if minor < min_minor_axis_px or minor > max_minor_axis_px:
            continue
        if cy > y_limit:
            continue

        plausible.append((props, cnt, area))

    if len(plausible) < 2:
        logger.error(
            "Fewer than 2 plausible drop candidates after geometry filter "
            "(found %d). Verify size thresholds or that both drops are visible.",
            len(plausible),
        )
        return None

    # Split candidates into left half and right half of the frame
    left_candidates  = [(p, c, a) for p, c, a in plausible if p["center_x"] <  w / 2]
    right_candidates = [(p, c, a) for p, c, a in plausible if p["center_x"] >= w / 2]

    if left_candidates and right_candidates:
        # Preferred path: one winner per spatial half (largest area wins)
        best_left  = max(left_candidates,  key=lambda t: t[2])
        best_right = max(right_candidates, key=lambda t: t[2])
        left_props,  left_cnt  = best_left[0],  best_left[1]
        right_props, right_cnt = best_right[0], best_right[1]
    else:
        # Fallback: take the two globally largest plausible candidates
        logger.warning(
            "Candidates found only on one side of frame "
            "(left=%d, right=%d). Using global top-2 fallback.",
            len(left_candidates), len(right_candidates),
        )
        top_two = sorted(plausible, key=lambda t: t[2], reverse=True)[:2]
        # Sort by center_x so left→control, right→sample rule still applies
        top_two.sort(key=lambda t: t[0]["center_x"])
        left_props,  left_cnt  = top_two[0][0], top_two[0][1]
        right_props, right_cnt = top_two[1][0], top_two[1][1]

    # ------------------------------------------------------------------
    # Step 5 — Debug log for selected candidates
    # ------------------------------------------------------------------
    logger.debug(
        "Selected LEFT  (control): center=(%.1f, %.1f)  major=%.1f  minor=%.1f  area=%.1f px²",
        left_props["center_x"], left_props["center_y"],
        left_props["major_axis"], left_props["minor_axis"],
        cv2.contourArea(left_cnt),
    )
    logger.debug(
        "Selected RIGHT (sample) : center=(%.1f, %.1f)  major=%.1f  minor=%.1f  area=%.1f px²",
        right_props["center_x"], right_props["center_y"],
        right_props["major_axis"], right_props["minor_axis"],
        cv2.contourArea(right_cnt),
    )

    # ------------------------------------------------------------------
    # Step 6 — Spatial classification (unbreakable rule)
    #
    #   left  candidate (smaller centroid X) → label = 'control'
    #   right candidate (larger  centroid X) → label = 'sample'
    # ------------------------------------------------------------------
    logger.debug(
        "Spatial classification: control centroid_x=%.1f | sample centroid_x=%.1f",
        left_props["center_x"],
        right_props["center_x"],
    )

    # ------------------------------------------------------------------
    # Step 7 — Build BubbleDetection objects with physical conversion
    # ------------------------------------------------------------------
    detection_control = _build_detection(
        left_props,  left_cnt,  px_to_mm, label="control"
    )
    detection_sample  = _build_detection(
        right_props, right_cnt, px_to_mm, label="sample"
    )

    return {
        "control": detection_control,
        "sample":  detection_sample,
    }


# ---------------------------------------------------------------------------
# Legacy single-drop shim (kept for backward compatibility with scripts that
# still call detect_bubble).  New code should call detect_bubbles() instead.
# ---------------------------------------------------------------------------

def detect_bubble(
    frame: np.ndarray,
    px_to_mm: float | None = None,
    blur_kernel: int = 7,
    clip_limit: float = 3.0,
    **_kwargs,
) -> "BubbleDetection | None":
    """Backward-compatible wrapper around :func:`detect_bubbles`.

    .. deprecated::
        This function is retained so that existing scripts (``analyze_image.py``,
        ``analyze_video.py``) do not break immediately.  It returns only the
        *sample* drop detection.  Migrate callers to :func:`detect_bubbles`
        to access both drops.

    Args:
        frame      : BGR colour image.
        px_to_mm   : Calibration factor in px/mm.
        blur_kernel: Gaussian blur kernel size.
        clip_limit : CLAHE clip limit.
        **_kwargs  : Ignored; accepted for API compatibility.

    Returns:
        The *sample* :class:`BubbleDetection`, or None if detection fails.
    """
    import warnings
    warnings.warn(
        "detect_bubble() is deprecated. Use detect_bubbles() to access "
        "both 'control' and 'sample' BubbleDetection objects.",
        DeprecationWarning,
        stacklevel=2,
    )
    result = detect_bubbles(frame, px_to_mm=px_to_mm,
                            blur_kernel=blur_kernel, clip_limit=clip_limit)
    if result is None:
        return None
    return result.get("sample")
