"""
Detection module — Dual pendant-drop detection with spatial classification.

Pipeline overview:
    1. Preprocessing   : Grayscale conversion + CLAHE contrast enhancement +
                         Gaussian blur applied per ROI.
    2. Hough coarse    : cv2.HoughCircles locates drop centre candidates
                         inside each fixed ROI.
    3. Candidate scoring : For each Hough circle, a dynamic ROI is cropped,
                         Otsu + MORPH_OPEN is applied, and an ellipse is fitted.
                         Candidates are filtered by axis size, centre position,
                         and ROI-edge proximity.  The surviving candidate with
                         the highest score (area − 2·distance_to_expected) is
                         selected.
    4. Global coords   : Ellipse centre and contour are translated back to
                         full-frame coordinates.
    5. Spatial rule    : Left ROI → 'control' | Right ROI → 'sample'.
    6. Physical conv.  : Semi-axes → mm → spheroid geometry (if px_to_mm).

Returns
-------
dict[str, BubbleDetection | None]
    {'control': BubbleDetection | None, 'sample': BubbleDetection | None}

    Returns None for an individual key if that drop could not be detected.
    Returns None (the whole dict) only if either ROI fails.
"""

from __future__ import annotations

import logging
import math
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
# Morphological constants for capillary isolation (used inside dynamic ROI)
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
# Internal helpers — geometry / masking
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

    An elliptical structuring element of size ``_CAPILLARY_KERNEL_SIZE`` x
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
        label         : Spatial label --- 'control' or 'sample'.

    Returns:
        Fully populated :class:`BubbleDetection` instance.
    """
    result = BubbleDetection()
    result.label = label
    result.method = "hough+morphopen+ellipse"
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
# Hough + dynamic-ROI ellipse detector (one drop per call)
# ---------------------------------------------------------------------------

def _detect_drop_in_roi(
    frame: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    roi_label: str,
    frame_h: int,
    blur_kernel: int = 7,
    clip_limit: float = 3.0,
    min_radius: int = 12,
    max_radius: int = 80,
    hough_dp: float = 1.2,
    hough_param1: float = 100.0,
    hough_param2: float = 20.0,
    margin_factor: float = 1.7,
    min_major: float = 35.0,
    max_major: float = 125.0,
    min_minor: float = 25.0,
    max_minor: float = 100.0,
) -> "tuple[dict, np.ndarray] | None":
    """Detect one pendant drop: Hough coarse localisation + ellipse fine fit.

    All Hough circle candidates are evaluated via ellipse fitting.  Candidates
    are filtered by:
        - axis size limits (min/max major & minor),
        - centre-y position in global frame coordinates,
        - ellipse proximity to ROI edges.

    The surviving candidate with the highest *score* is selected:
        score = ellipse_area_px − 2.0 × distance_to_expected_centre

    where expected centre = (mid-x of ROI, y0 + 0.48 × ROI height).

    Steps
    -----
    1. Crop the fixed ROI from ``frame``.
    2. Preprocess (grayscale + CLAHE + Gaussian blur).
    3. Run ``cv2.HoughCircles`` to locate drop centre candidates.
    4. For each Hough circle: build a tight dynamic crop, apply Otsu binary
       mask + MORPH_OPEN, find largest contour, fit ellipse.
    5. Filter candidates (size, centre position, edge proximity).
    6. Score and select best candidate.
    7. Translate ellipse centre and contour back to **global** frame coords.

    Args:
        frame         : Full BGR frame (uint8, 3-channel).
        x0, y0        : Top-left corner of the fixed ROI (global px).
        x1, y1        : Bottom-right corner (exclusive) of the ROI (global px).
        roi_label     : Human-readable label for log messages.
        frame_h       : Full-frame height in pixels (for centre-y checks).
        blur_kernel   : Gaussian kernel size for preprocessing.
        clip_limit    : CLAHE clip limit for preprocessing.
        min_radius    : Minimum Hough circle radius (px).
        max_radius    : Maximum Hough circle radius (px).
        hough_dp      : Inverse ratio of accumulator resolution.
        hough_param1  : Canny high threshold for Hough.
        hough_param2  : Accumulator threshold for Hough circle centres.
        margin_factor : Multiplier applied to Hough radius for the dynamic crop.
        min_major     : Minimum accepted major axis of the fitted ellipse (px).
        max_major     : Maximum accepted major axis of the fitted ellipse (px).
        min_minor     : Minimum accepted minor axis of the fitted ellipse (px).
        max_minor     : Maximum accepted minor axis of the fitted ellipse (px).

    Returns:
        ``(ellipse_props_global, global_contour)`` on success, or ``None`` if
        Hough finds no valid candidate after filtering and scoring.
    """
    h = frame_h

    # Centre-y acceptance window in global frame coordinates
    cy_min_global = int(0.06 * h)
    cy_max_global = int(0.32 * h)

    # Expected drop centre for scoring (global frame coords)
    expected_x = (x0 + x1) / 2.0
    expected_y = y0 + 0.48 * (y1 - y0)

    # ROI edge rejection margin (px)
    _EDGE_MARGIN = 3

    # 1. Crop fixed ROI
    frame_roi = frame[y0:y1, x0:x1]

    # 2. Preprocess (grayscale + CLAHE + Gaussian blur)
    gray_roi = preprocess(frame_roi, blur_kernel=blur_kernel, clip_limit=clip_limit)

    # 3. Hough Circle Transform
    circles = cv2.HoughCircles(
        gray_roi,
        cv2.HOUGH_GRADIENT,
        dp=hough_dp,
        minDist=min_radius * 2,
        param1=hough_param1,
        param2=hough_param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        logger.debug("ROI [%s]: Hough found 0 circles. Detection failed.", roi_label)
        return None

    circles_round = np.round(circles[0]).astype(int)
    n_circles = len(circles_round)
    logger.debug("ROI [%s]: Hough found %d circle(s).", roi_label, n_circles)

    roi_h, roi_w = gray_roi.shape[:2]

    # ------------------------------------------------------------------
    # 4–5. Evaluate every Hough candidate
    # ------------------------------------------------------------------
    candidates: list[tuple[float, dict, np.ndarray]] = []  # (score, props_global, global_cnt)
    rejected_shape: int = 0

    for idx, circle in enumerate(circles_round):
        hcx, hcy, hr = int(circle[0]), int(circle[1]), int(circle[2])

        # 4. Dynamic crop around this Hough circle
        margin = int(hr * margin_factor)
        dx0 = max(0, hcx - margin)
        dy0 = max(0, hcy - margin)
        dx1 = min(roi_w, hcx + margin)
        dy1 = min(roi_h, hcy + margin)

        if dx1 <= dx0 or dy1 <= dy0:
            logger.debug(
                "ROI [%s] circle #%d: dynamic crop degenerate (%d,%d)-(%d,%d). Skip.",
                roi_label, idx, dx0, dy0, dx1, dy1,
            )
            continue

        gray_dyn = gray_roi[dy0:dy1, dx0:dx1]

        # Binary mask + MORPH_OPEN + largest contour + ellipse fit
        mask_dyn = _binary_mask(gray_dyn)
        mask_dyn = _isolate_drops(mask_dyn)

        contours, _ = cv2.findContours(
            mask_dyn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            logger.debug(
                "ROI [%s] circle #%d: no contours inside dynamic crop.", roi_label, idx
            )
            continue

        best_cnt = max(contours, key=cv2.contourArea)
        props = _fit_single_ellipse(best_cnt)

        if props is None:
            logger.debug(
                "ROI [%s] circle #%d: contour < 5 pts; ellipse fit skipped.",
                roi_label, idx,
            )
            continue

        # Translate to global frame coords (ROI offset + dynamic-crop offset)
        gx = x0 + dx0
        gy = y0 + dy0

        props_g = dict(props)
        props_g["center_x"] += gx
        props_g["center_y"] += gy
        global_cnt = best_cnt + np.array([gx, gy], dtype=np.int32)

        center_x_g = props_g["center_x"]
        center_y_g = props_g["center_y"]
        major      = props_g["major_axis"]
        minor      = props_g["minor_axis"]
        semi_major = props_g["semi_major"]
        semi_minor = props_g["semi_minor"]

        # ---- Filter 1: axis size limits -----------------------------------
        if major < min_major:
            logger.debug(
                "ROI [%s] circle #%d: major=%.1f < min %.1f — rejected (too small).",
                roi_label, idx, major, min_major,
            )
            continue
        if major > max_major:
            logger.debug(
                "ROI [%s] circle #%d: major=%.1f > max %.1f — rejected (too large).",
                roi_label, idx, major, max_major,
            )
            continue
        if minor < min_minor:
            logger.debug(
                "ROI [%s] circle #%d: minor=%.1f < min %.1f — rejected (too small).",
                roi_label, idx, minor, min_minor,
            )
            continue
        if minor > max_minor:
            logger.debug(
                "ROI [%s] circle #%d: minor=%.1f > max %.1f — rejected (too large).",
                roi_label, idx, minor, max_minor,
            )
            continue

        # ---- Filter 2: centre-y position in global frame ------------------
        if center_y_g <= cy_min_global:
            logger.debug(
                "ROI [%s] circle #%d: center_y=%.1f <= %d — rejected (too high).",
                roi_label, idx, center_y_g, cy_min_global,
            )
            continue
        if center_y_g >= cy_max_global:
            logger.debug(
                "ROI [%s] circle #%d: center_y=%.1f >= %d — rejected (too low).",
                roi_label, idx, center_y_g, cy_max_global,
            )
            continue

        # ---- Filter 3: ellipse must not be clipped by ROI edges -----------
        if center_x_g - semi_major < x0 + _EDGE_MARGIN:
            logger.debug(
                "ROI [%s] circle #%d: ellipse clips left ROI edge — rejected.",
                roi_label, idx,
            )
            continue
        if center_x_g + semi_major > x1 - _EDGE_MARGIN:
            logger.debug(
                "ROI [%s] circle #%d: ellipse clips right ROI edge — rejected.",
                roi_label, idx,
            )
            continue
        if center_y_g - semi_minor < y0 + _EDGE_MARGIN:
            logger.debug(
                "ROI [%s] circle #%d: ellipse clips top ROI edge — rejected.",
                roi_label, idx,
            )
            continue
        if center_y_g + semi_minor > y1 - _EDGE_MARGIN:
            logger.debug(
                "ROI [%s] circle #%d: ellipse clips bottom ROI edge — rejected.",
                roi_label, idx,
            )
            continue

        # ---- Filter 4: shape plausibility (reject extreme elongation) ------
        axis_ratio  = minor / major          # in (0, 1]; closer to 1 = rounder
        ecc         = props_g["eccentricity"]

        if axis_ratio < 0.65:
            logger.debug(
                "ROI [%s] circle #%d: major=%.1f minor=%.1f "
                "axis_ratio=%.3f < 0.65  ecc=%.3f — rejected (shape).",
                roi_label, idx, major, minor, axis_ratio, ecc,
            )
            rejected_shape += 1
            continue
        if ecc > 0.78:
            logger.debug(
                "ROI [%s] circle #%d: major=%.1f minor=%.1f "
                "axis_ratio=%.3f  ecc=%.3f > 0.78 — rejected (shape).",
                roi_label, idx, major, minor, axis_ratio, ecc,
            )
            rejected_shape += 1
            continue

        # ---- Compute score ------------------------------------------------
        area_px    = math.pi * semi_major * semi_minor
        distance   = math.sqrt(
            (center_x_g - expected_x) ** 2 + (center_y_g - expected_y) ** 2
        )
        shape_score = axis_ratio          # in (0.50, 1]; favours rounder drops
        score = area_px - 2.0 * distance + 500.0 * shape_score

        logger.debug(
            "ROI [%s] circle #%d accepted: "
            "center=(%.1f, %.1f)  major=%.1f  minor=%.1f  "
            "axis_ratio=%.3f  ecc=%.3f  "
            "area_px=%.1f  dist=%.1f  score=%.2f",
            roi_label, idx,
            center_x_g, center_y_g, major, minor,
            axis_ratio, ecc,
            area_px, distance, score,
        )

        candidates.append((score, props_g, global_cnt))

    # ------------------------------------------------------------------
    # 6. Select best candidate
    # ------------------------------------------------------------------
    n_passed = len(candidates)
    logger.debug(
        "ROI [%s]: %d / %d Hough circle(s) passed all filters "
        "(%d rejected by shape).",
        roi_label, n_passed, n_circles, rejected_shape,
    )

    if not candidates:
        return None

    best_score, best_props, best_cnt_global = max(candidates, key=lambda t: t[0])

    logger.debug(
        "ROI [%s] selected: center=(%.1f, %.1f)  major=%.1f  minor=%.1f  score=%.2f",
        roi_label,
        best_props["center_x"], best_props["center_y"],
        best_props["major_axis"], best_props["minor_axis"],
        best_score,
    )

    return best_props, best_cnt_global



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
    1. Two fixed, tightened ROIs are defined (left → control, right → sample).
    2. Inside each ROI: preprocess → Hough coarse localisation → per-candidate
       dynamic crop → Otsu + MORPH_OPEN → largest contour → ellipse fit →
       axis / position / edge checks → scored selection.
    3. Ellipse centre and contour are translated back to global frame coords.
    4. _build_detection populates a BubbleDetection with optional mm conversion.

    Spatial rule (unbreakable)
    --------------------------
    * Left  ROI (x: 8–42 % of frame width,  y: 0–36 %)  → label = 'control'
    * Right ROI (x: 46–78 % of frame width, y: 0–36 %)  → label = 'sample'

    Args:
        frame      : BGR colour image from the microscope (uint8, 3-channel).
        px_to_mm   : Calibration factor in pixels per millimetre.
                     When None, only pixel-space measurements are populated.
        blur_kernel: Gaussian blur kernel size (must be a positive odd integer).
        clip_limit : CLAHE clip limit for contrast enhancement.

    Returns:
        ``{'control': BubbleDetection, 'sample': BubbleDetection}``

        Returns ``None`` if either ROI fails to produce a valid detection.

    Raises:
        ValueError: If ``frame`` is not a 3-channel BGR image.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"Expected a 3-channel BGR image; got shape {frame.shape}."
        )

    h, w = frame.shape[:2]

    # ------------------------------------------------------------------
    # Hough + ellipse parameters — tune here if magnification changes
    # ------------------------------------------------------------------
    _MIN_RADIUS:    int   = 12
    _MAX_RADIUS:    int   = 80
    _HOUGH_DP:      float = 1.2
    _HOUGH_PARAM1:  float = 100.0
    _HOUGH_PARAM2:  float = 20.0
    _MARGIN_FACTOR: float = 1.7
    _MIN_MAJOR:     float = 35.0
    _MAX_MAJOR:     float = 125.0
    _MIN_MINOR:     float = 25.0
    _MAX_MINOR:     float = 100.0

    # ------------------------------------------------------------------
    # Fixed ROIs — tightened to avoid background false positives
    # ------------------------------------------------------------------
    # CONTROL drop — left capillary
    c_x0 = int(0.08 * w);  c_x1 = int(0.42 * w)
    c_y0 = int(0.00 * h);  c_y1 = int(0.36 * h)

    # SAMPLE drop — right capillary (right edge pulled in to 78 % to avoid
    # spurious detections in the far upper-right corner of the frame)
    s_x0 = int(0.46 * w);  s_x1 = int(0.78 * w)
    s_y0 = int(0.00 * h);  s_y1 = int(0.36 * h)

    # ------------------------------------------------------------------
    # Hough + ellipse detection per ROI
    # ------------------------------------------------------------------
    control_result = _detect_drop_in_roi(
        frame,
        c_x0, c_y0, c_x1, c_y1,
        roi_label="control",
        frame_h=h,
        blur_kernel=blur_kernel,
        clip_limit=clip_limit,
        min_radius=_MIN_RADIUS,
        max_radius=_MAX_RADIUS,
        hough_dp=_HOUGH_DP,
        hough_param1=_HOUGH_PARAM1,
        hough_param2=_HOUGH_PARAM2,
        margin_factor=_MARGIN_FACTOR,
        min_major=_MIN_MAJOR,
        max_major=_MAX_MAJOR,
        min_minor=_MIN_MINOR,
        max_minor=_MAX_MINOR,
    )

    sample_result = _detect_drop_in_roi(
        frame,
        s_x0, s_y0, s_x1, s_y1,
        roi_label="sample",
        frame_h=h,
        blur_kernel=blur_kernel,
        clip_limit=clip_limit,
        min_radius=_MIN_RADIUS,
        max_radius=_MAX_RADIUS,
        hough_dp=_HOUGH_DP,
        hough_param1=_HOUGH_PARAM1,
        hough_param2=_HOUGH_PARAM2,
        margin_factor=_MARGIN_FACTOR,
        min_major=_MIN_MAJOR,
        max_major=_MAX_MAJOR,
        min_minor=_MIN_MINOR,
        max_minor=_MAX_MINOR,
    )

    if control_result is None:
        logger.error(
            "ROI [control]: detection failed — no Hough circle survived the "
            "size / position / edge filters. "
            "Check ROI bounds, Hough parameters, or size thresholds."
        )
        return None

    if sample_result is None:
        logger.error(
            "ROI [sample]: detection failed — no Hough circle survived the "
            "size / position / edge filters. "
            "Check ROI bounds, Hough parameters, or size thresholds."
        )
        return None

    left_props,  left_cnt  = control_result
    right_props, right_cnt = sample_result

    # ------------------------------------------------------------------
    # Spatial classification (unbreakable rule)
    #
    #   left  ROI → label = 'control'
    #   right ROI → label = 'sample'
    # ------------------------------------------------------------------
    logger.debug(
        "Spatial classification: control centroid_x=%.1f | sample centroid_x=%.1f",
        left_props["center_x"],
        right_props["center_x"],
    )

    # ------------------------------------------------------------------
    # Build BubbleDetection objects with physical conversion
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
