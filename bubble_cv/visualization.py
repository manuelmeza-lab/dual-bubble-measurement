"""
Visualization module — Detection overlay and dual time-series plots.

Provides functions to draw fitted ellipses on frames for visual verification
and to generate comparative time-series plots of both pendant-drop evolutions.

Public API
----------
draw_detection(frame, detection, ...)
    Draw a single detection overlay on a frame.

draw_detection_dual(frame, control, sample, ...)
    Draw control and sample overlays simultaneously with distinct colors.

save_annotated_frame(frame, detection, path, ...)
    Save a single-detection annotated frame.

save_annotated_frame_dual(frame, control, sample, path, ...)
    Save a dual-detection annotated frame.

plot_timeseries(csv_path, ...)          [legacy — single-drop CSV]
plot_dual_timeseries(csv_path, ...)     [dual CSV with prefixed columns]
plot_binned_timeseries(...)             [legacy]
plot_binned_dual_timeseries(...)        [dual binned CSV]
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bubble_cv.detection import BubbleDetection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette (BGR for OpenCV; hex/rgb for Matplotlib)
# ---------------------------------------------------------------------------

# Single-drop legacy colors
COLOR_ELLIPSE   = (0, 255, 0)       # Green
COLOR_CENTER    = (0, 0, 255)       # Red
COLOR_AXES      = (255, 255, 0)     # Cyan
COLOR_TEXT_BG   = (0, 0, 0)         # Black
COLOR_TEXT_FG   = (255, 255, 255)   # White

# Dual-drop colors — control=blue, sample=orange (colorblind-safe pair)
_DUAL_COLORS_BGR = {
    "control": (220, 130, 0),    # Deep sky blue (BGR)
    "sample":  (0, 140, 255),    # Orange (BGR)
}
_DUAL_COLORS_HEX = {
    "control": "#1565C0",        # Material Blue 800
    "sample":  "#E65100",        # Material Deep-Orange 900
}
_DUAL_MARKERS = {
    "control": "o",
    "sample":  "s",
}


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_detection(
    frame: np.ndarray,
    detection: BubbleDetection,
    show_measurements: bool = True,
    thickness: int = 2,
    color_ellipse: tuple[int, int, int] = COLOR_ELLIPSE,
    text_y_offset: int = 30,
) -> np.ndarray:
    """Draw the fitted ellipse and measurements on a frame (single drop).

    Args:
        frame: BGR image to annotate.
        detection: BubbleDetection result to visualize.
        show_measurements: Whether to display text measurements.
        thickness: Line thickness for the ellipse and axes.
        color_ellipse: BGR color for the ellipse outline.
        text_y_offset: Starting Y position for annotation text.

    Returns:
        Annotated BGR image (copy of original frame).
    """
    annotated = frame if frame.flags["OWNDATA"] is False else frame.copy()
    cx = int(detection.center_x_px)
    cy = int(detection.center_y_px)
    major = int(detection.major_axis_px)
    minor = int(detection.minor_axis_px)
    angle = detection.angle_deg

    # Fitted ellipse
    cv2.ellipse(
        annotated,
        (cx, cy),
        (max(1, major // 2), max(1, minor // 2)),
        angle, 0, 360,
        color_ellipse, thickness,
    )

    # Center crosshair
    cross_size = 10
    cv2.line(annotated, (cx - cross_size, cy), (cx + cross_size, cy), COLOR_CENTER, 1)
    cv2.line(annotated, (cx, cy - cross_size), (cx, cy + cross_size), COLOR_CENTER, 1)
    cv2.circle(annotated, (cx, cy), 3, COLOR_CENTER, -1)

    # Axis lines
    angle_rad = np.radians(angle)
    dx_maj = int((major / 2) * np.cos(angle_rad))
    dy_maj = int((major / 2) * np.sin(angle_rad))
    cv2.line(annotated, (cx - dx_maj, cy - dy_maj), (cx + dx_maj, cy + dy_maj), COLOR_AXES, 1)
    dx_min = int((minor / 2) * np.cos(angle_rad + np.pi / 2))
    dy_min = int((minor / 2) * np.sin(angle_rad + np.pi / 2))
    cv2.line(annotated, (cx - dx_min, cy - dy_min), (cx + dx_min, cy + dy_min), COLOR_AXES, 1)

    # Text overlay
    if show_measurements:
        label_str = f"[{detection.label}]" if detection.label else ""
        lines = [
            f"{label_str} {detection.method}",
            f"Axes: {detection.major_axis_px:.0f}x{detection.minor_axis_px:.0f} px",
            f"Ecc: {detection.eccentricity:.3f}",
        ]
        if detection.equiv_diameter_mm is not None:
            lines.append(f"D_eq: {detection.equiv_diameter_mm:.3f} mm")
        if detection.volume_mm3 is not None:
            lines.append(f"Vol: {detection.volume_mm3:.4f} mm³")

        y = text_y_offset
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (10, y - th - 3), (14 + tw, y + 4), COLOR_TEXT_BG, -1)
            cv2.putText(
                annotated, line, (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_ellipse, 1, cv2.LINE_AA,
            )
            y += th + 10

    return annotated


def draw_detection_dual(
    frame: np.ndarray,
    control: BubbleDetection,
    sample: BubbleDetection,
    show_measurements: bool = True,
    thickness: int = 2,
) -> np.ndarray:
    """Draw both pendant-drop detections on a single frame.

    Both drops share the same green ellipse color so the overlay is
    consistent with the single-drop convention.  The text blocks are
    spatially separated to avoid overlap:

    * **[CONTROL]** block → anchored to the **upper-left** corner.
    * **[MUESTRA]**  block → anchored to the **upper-right** corner
      (right-aligned so it never spills off-screen).

    Each ellipse is drawn with:
        - Green contour (``COLOR_ELLIPSE``).
        - Green major/minor axis lines (``COLOR_AXES``).
        - Red center crosshair (``COLOR_CENTER``).

    Args:
        frame: BGR image to annotate (will be deep-copied).
        control: Detection for the left (control) drop.
        sample: Detection for the right (sample) drop.
        show_measurements: Whether to render metric text blocks.
        thickness: Line thickness for ellipses and axis lines.

    Returns:
        Annotated BGR image (deep copy — original is never modified).
    """
    annotated = frame.copy()
    img_h, img_w = annotated.shape[:2]

    # ------------------------------------------------------------------
    # Internal helper: draw one ellipse + crosshair + axes
    # ------------------------------------------------------------------
    def _draw_ellipse_geometry(img: np.ndarray, det: BubbleDetection) -> None:
        cx = int(det.center_x_px)
        cy = int(det.center_y_px)
        major = int(det.major_axis_px)
        minor = int(det.minor_axis_px)
        angle = det.angle_deg

        # Fitted ellipse — green
        cv2.ellipse(
            img, (cx, cy),
            (max(1, major // 2), max(1, minor // 2)),
            angle, 0, 360,
            COLOR_ELLIPSE, thickness,
        )

        # Center crosshair — red
        cs = 10
        cv2.line(img, (cx - cs, cy), (cx + cs, cy), COLOR_CENTER, 1)
        cv2.line(img, (cx, cy - cs), (cx, cy + cs), COLOR_CENTER, 1)
        cv2.circle(img, (cx, cy), 3, COLOR_CENTER, -1)

        # Axis lines — cyan
        rad = np.radians(angle)
        dx_maj = int((major / 2) * np.cos(rad))
        dy_maj = int((major / 2) * np.sin(rad))
        cv2.line(img, (cx - dx_maj, cy - dy_maj), (cx + dx_maj, cy + dy_maj), COLOR_AXES, 1)
        dx_min = int((minor / 2) * np.cos(rad + np.pi / 2))
        dy_min = int((minor / 2) * np.sin(rad + np.pi / 2))
        cv2.line(img, (cx - dx_min, cy - dy_min), (cx + dx_min, cy + dy_min), COLOR_AXES, 1)

    # ------------------------------------------------------------------
    # Internal helper: build metric lines for one drop
    # ------------------------------------------------------------------
    def _metric_lines(det: BubbleDetection, header: str) -> list[str]:
        lines = [
            header,
            f"Metodo: {det.method}",
            f"Ejes: {det.major_axis_px:.0f} x {det.minor_axis_px:.0f} px",
            f"Exc: {det.eccentricity:.3f}",
        ]
        if det.equiv_diameter_mm is not None:
            lines.append(f"D_eq: {det.equiv_diameter_mm:.3f} mm")
        if det.volume_mm3 is not None:
            lines.append(f"Vol: {det.volume_mm3:.4f} mm3")
        return lines

    # ------------------------------------------------------------------
    # Internal helper: render a left-anchored text block
    # ------------------------------------------------------------------
    _FONT      = cv2.FONT_HERSHEY_SIMPLEX
    _FONT_SCALE = 0.52
    _FONT_THICK = 1
    _LINE_PAD   = 8    # px between lines
    _MARGIN     = 10   # px from image border

    def _draw_text_left(
        img: np.ndarray,
        lines: list[str],
        color_fg: tuple[int, int, int],
    ) -> None:
        """Render lines left-aligned from the top-left corner."""
        y = _MARGIN + 18   # first baseline
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, _FONT, _FONT_SCALE, _FONT_THICK)
            x0, y0 = _MARGIN, y - th - 3
            x1, y1 = _MARGIN + tw + 6, y + 4
            # Clip to image bounds
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(img_w - 1, x1), min(img_h - 1, y1)
            cv2.rectangle(img, (x0, y0), (x1, y1), COLOR_TEXT_BG, -1)
            cv2.putText(
                img, line, (_MARGIN + 3, y),
                _FONT, _FONT_SCALE, color_fg, _FONT_THICK, cv2.LINE_AA,
            )
            y += th + _LINE_PAD

    # ------------------------------------------------------------------
    # Internal helper: render a right-anchored text block
    # ------------------------------------------------------------------
    def _draw_text_right(
        img: np.ndarray,
        lines: list[str],
        color_fg: tuple[int, int, int],
    ) -> None:
        """Render lines right-aligned from the top-right corner."""
        y = _MARGIN + 18
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, _FONT, _FONT_SCALE, _FONT_THICK)
            # Right edge: img_w - _MARGIN; text starts at img_w - _MARGIN - tw
            x_text = img_w - _MARGIN - tw - 3
            x0, y0 = x_text - 3, y - th - 3
            x1, y1 = img_w - _MARGIN, y + 4
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(img_w - 1, x1), min(img_h - 1, y1)
            cv2.rectangle(img, (x0, y0), (x1, y1), COLOR_TEXT_BG, -1)
            cv2.putText(
                img, line, (max(0, x_text), y),
                _FONT, _FONT_SCALE, color_fg, _FONT_THICK, cv2.LINE_AA,
            )
            y += th + _LINE_PAD

    # ------------------------------------------------------------------
    # Step 1 — draw both ellipses (geometry only)
    # ------------------------------------------------------------------
    _draw_ellipse_geometry(annotated, control)
    _draw_ellipse_geometry(annotated, sample)

    # ------------------------------------------------------------------
    # Step 2 — draw text blocks (if requested)
    # ------------------------------------------------------------------
    if show_measurements:
        ctrl_color = _DUAL_COLORS_BGR["control"]   # blue tint for [CONTROL]
        samp_color = _DUAL_COLORS_BGR["sample"]    # orange tint for [MUESTRA]

        ctrl_lines = _metric_lines(control, "[CONTROL]")
        samp_lines = _metric_lines(sample,  "[MUESTRA]")

        _draw_text_left(annotated,  ctrl_lines, ctrl_color)
        _draw_text_right(annotated, samp_lines, samp_color)

    return annotated


# ---------------------------------------------------------------------------
# Frame-save wrappers
# ---------------------------------------------------------------------------

def save_annotated_frame(
    frame: np.ndarray,
    detection: BubbleDetection,
    output_path: str | Path,
    show_measurements: bool = True,
) -> None:
    """Save a single-detection annotated frame to disk.

    Args:
        frame: Original BGR image.
        detection: Detection result to overlay.
        output_path: Output file path (.png recommended).
        show_measurements: Whether to display text measurements.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = draw_detection(frame.copy(), detection, show_measurements)
    cv2.imwrite(str(output_path), annotated)
    logger.info("Saved annotated frame: %s", output_path)


def save_annotated_frame_dual(
    frame: np.ndarray,
    control: BubbleDetection,
    sample: BubbleDetection,
    output_path: str | Path,
    show_measurements: bool = True,
) -> None:
    """Save a dual-detection annotated frame to disk.

    Both drops are rendered with distinct colors and labels.

    Args:
        frame: Original BGR image.
        control: Left (control) drop detection.
        sample: Right (sample) drop detection.
        output_path: Output file path (.png recommended).
        show_measurements: Whether to display text measurements.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = draw_detection_dual(frame, control, sample, show_measurements)
    cv2.imwrite(str(output_path), annotated)
    logger.info("Saved dual annotated frame: %s", output_path)


# ---------------------------------------------------------------------------
# Internal plot utilities
# ---------------------------------------------------------------------------

_RCPARAMS = {
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _get_time_axis(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """Return the appropriate x-axis series and label for a DataFrame."""
    if "timestamp_s" in df.columns:
        return df["timestamp_s"], "Time (s)"
    if "frame_id" in df.columns:
        return df["frame_id"], "Frame"
    return pd.Series(df.index), "Sample"


def _save_fig(fig: plt.Figure, path: Path, show: bool) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Saved plot: %s", path)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dual time-series plots (new — primary API)
# ---------------------------------------------------------------------------

def plot_dual_timeseries(
    csv_path: str | Path,
    output_dir: str | Path | None = None,
    show: bool = False,
) -> None:
    """Generate comparative time-series plots from a dual-drop CSV file.

    Each plot overlays the 'control' and 'sample' curves on the same axes
    with a clear legend, facilitating visual comparison of evaporation rates.

    Plots generated:
        1. ``diameter_vs_time_dual.png``   — D_eq (mm or px) vs time
        2. ``volume_vs_time_dual.png``     — Volume (mm³) vs time
        3. ``eccentricity_vs_time_dual.png`` — Eccentricity vs time
        4. ``evap_rate_vs_time_dual.png``  — dV/dt (mm³/s) vs time
        5. ``radius_squared_vs_time_dual.png`` — r²_eq (mm²) vs time

    Args:
        csv_path: Path to the paired CSV produced by analyze_video.py.
        output_dir: Directory to save plots.  Defaults to same dir as CSV.
        show: If True, display plots interactively.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.error("CSV file not found: %s", csv_path)
        return

    df = pd.read_csv(csv_path)
    output_dir = Path(output_dir) if output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    x, x_label = _get_time_axis(df)
    plt.rcParams.update(_RCPARAMS)

    LABELS = ("control", "sample")

    # ------------------------------------------------------------------
    # Helper: build per-label valid/invalid masks
    # ------------------------------------------------------------------
    def _masks(lbl: str) -> tuple[pd.Series, pd.Series]:
        col = f"{lbl}_tracking_valid"
        if col in df.columns:
            v = df[col] == True
            return v, ~v
        return pd.Series([True] * len(df)), pd.Series([False] * len(df))

    # ------------------------------------------------------------------
    # Plot 1 — Equivalent diameter
    # ------------------------------------------------------------------
    has_mm = all(f"{lbl}_equiv_diameter_mm" in df.columns for lbl in LABELS)
    y_key  = "equiv_diameter_mm" if has_mm else "equiv_diameter_px"
    y_unit = "mm" if has_mm else "px"

    fig, ax = plt.subplots(figsize=(11, 5))
    for lbl in LABELS:
        col = f"{lbl}_{y_key}"
        if col not in df.columns:
            continue
        valid, invalid = _masks(lbl)
        color = _DUAL_COLORS_HEX[lbl]
        marker = _DUAL_MARKERS[lbl]
        if valid.any():
            ax.plot(x[valid], df.loc[valid, col], "-", color=color, alpha=0.45)
            ax.scatter(
                x[valid], df.loc[valid, col],
                color=color, marker=marker, s=28, alpha=0.85,
                edgecolors="k", linewidths=0.4,
                label=f"{lbl} (valid)",
            )
        if invalid.any():
            ax.scatter(
                x[invalid], df.loc[invalid, col],
                color=color, marker="x", s=35, alpha=0.6,
                label=f"{lbl} (rejected)",
            )
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"Equivalent Diameter ({y_unit})")
    ax.set_title("Pendant Drop Diameter — Control vs Sample")
    ax.legend(framealpha=0.8)
    _save_fig(fig, output_dir / "diameter_vs_time_dual.png", show)

    # ------------------------------------------------------------------
    # Plot 2 — Volume
    # ------------------------------------------------------------------
    if all(f"{lbl}_volume_mm3" in df.columns for lbl in LABELS):
        fig, ax = plt.subplots(figsize=(11, 5))
        for lbl in LABELS:
            col = f"{lbl}_volume_mm3"
            valid, invalid = _masks(lbl)
            color = _DUAL_COLORS_HEX[lbl]
            marker = _DUAL_MARKERS[lbl]
            if valid.any():
                ax.plot(x[valid], df.loc[valid, col], "-", color=color, alpha=0.45)
                ax.scatter(
                    x[valid], df.loc[valid, col],
                    color=color, marker=marker, s=28, alpha=0.85,
                    edgecolors="k", linewidths=0.4,
                    label=f"{lbl} (valid)",
                )
            if invalid.any():
                ax.scatter(
                    x[invalid], df.loc[invalid, col],
                    color=color, marker="x", s=35, alpha=0.6,
                    label=f"{lbl} (rejected)",
                )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Volume (mm³)")
        ax.set_title("Pendant Drop Volume — Control vs Sample")
        ax.legend(framealpha=0.8)
        _save_fig(fig, output_dir / "volume_vs_time_dual.png", show)

    # ------------------------------------------------------------------
    # Plot 3 — Eccentricity
    # ------------------------------------------------------------------
    if all(f"{lbl}_eccentricity" in df.columns for lbl in LABELS):
        fig, ax = plt.subplots(figsize=(11, 5))
        for lbl in LABELS:
            col = f"{lbl}_eccentricity"
            valid, invalid = _masks(lbl)
            color = _DUAL_COLORS_HEX[lbl]
            marker = _DUAL_MARKERS[lbl]
            if valid.any():
                ax.plot(x[valid], df.loc[valid, col], "-", color=color, alpha=0.45)
                ax.scatter(
                    x[valid], df.loc[valid, col],
                    color=color, marker=marker, s=28, alpha=0.85,
                    edgecolors="k", linewidths=0.4,
                    label=f"{lbl}",
                )
            if invalid.any():
                ax.scatter(
                    x[invalid], df.loc[invalid, col],
                    color=color, marker="x", s=35, alpha=0.6,
                )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Eccentricity")
        ax.set_title("Pendant Drop Shape — Control vs Sample")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(framealpha=0.8)
        _save_fig(fig, output_dir / "eccentricity_vs_time_dual.png", show)

    # ------------------------------------------------------------------
    # Plot 4 — Evaporation rate dV/dt
    # ------------------------------------------------------------------
    evap_cols = [f"{lbl}_evap_rate_mm3_s" for lbl in LABELS]
    if any(c in df.columns and df[c].notna().any() for c in evap_cols):
        fig, ax = plt.subplots(figsize=(11, 5))
        for lbl in LABELS:
            col = f"{lbl}_evap_rate_mm3_s"
            if col not in df.columns:
                continue
            valid, _ = _masks(lbl)
            mask = valid & df[col].notna()
            color = _DUAL_COLORS_HEX[lbl]
            if mask.any():
                ax.plot(x[mask], df.loc[mask, col], "-", color=color, alpha=0.45)
                ax.scatter(
                    x[mask], df.loc[mask, col],
                    color=color, marker=_DUAL_MARKERS[lbl], s=28, alpha=0.85,
                    edgecolors="k", linewidths=0.4,
                    label=f"{lbl} dV/dt",
                )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Evaporation Rate (mm³/s)")
        ax.set_title("Instantaneous Evaporation Rate — Control vs Sample")
        ax.legend(framealpha=0.8)
        _save_fig(fig, output_dir / "evap_rate_vs_time_dual.png", show)

    # ------------------------------------------------------------------
    # Plot 5 — Radius squared r²_eq  (key for D² law)
    # ------------------------------------------------------------------
    r2_cols = [f"{lbl}_radius_eq_mm2" for lbl in LABELS]
    if any(c in df.columns and df[c].notna().any() for c in r2_cols):
        fig, ax = plt.subplots(figsize=(11, 5))
        for lbl in LABELS:
            col = f"{lbl}_radius_eq_mm2"
            if col not in df.columns:
                continue
            valid, invalid = _masks(lbl)
            color = _DUAL_COLORS_HEX[lbl]
            if valid.any():
                ax.plot(x[valid], df.loc[valid, col], "-", color=color, alpha=0.45)
                ax.scatter(
                    x[valid], df.loc[valid, col],
                    color=color, marker=_DUAL_MARKERS[lbl], s=28, alpha=0.85,
                    edgecolors="k", linewidths=0.4,
                    label=f"{lbl} $r_{{eq}}^2$",
                )
            if invalid.any():
                ax.scatter(
                    x[invalid], df.loc[invalid, col],
                    color=color, marker="x", s=35, alpha=0.6,
                )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Equivalent Radius Squared $r_{eq}^2$ (mm²)")
        ax.set_title("Radius Squared Evolution — Control vs Sample")
        ax.legend(framealpha=0.8)
        _save_fig(fig, output_dir / "radius_squared_vs_time_dual.png", show)


# ---------------------------------------------------------------------------
# Binned dual time-series plots (new)
# ---------------------------------------------------------------------------

def plot_binned_dual_timeseries(
    binned_csv_path: str | Path,
    output_dir: str | Path | None = None,
    show: bool = False,
) -> None:
    """Generate comparative binned plots with error bars (dual drops).

    Both control and sample curves are drawn on the same axes with
    distinct colors and error bars (±1 SD).

    Plots generated:
        1. ``diameter_binned_dual.png``
        2. ``volume_binned_dual.png``
        3. ``eccentricity_binned_dual.png``
        4. ``radius_squared_binned_dual.png``

    Args:
        binned_csv_path: Path to the binned CSV produced by analyze_video.py.
        output_dir: Output directory.  Defaults to same dir as CSV.
        show: If True, display plots interactively.
    """
    binned_csv_path = Path(binned_csv_path)
    if not binned_csv_path.exists():
        logger.error("Binned CSV not found: %s", binned_csv_path)
        return

    df = pd.read_csv(binned_csv_path)
    output_dir = Path(output_dir) if output_dir else binned_csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    x = df["time_mean_s"]
    x_label = "Time (s)"
    plt.rcParams.update(_RCPARAMS)

    LABELS = ("control", "sample")

    def _dual_errbar(ax: plt.Axes, base_key: str) -> bool:
        """Plot both drops on ax for the given base metric. Returns True if plotted."""
        plotted = False
        for lbl in LABELS:
            mean_col = f"{lbl}_{base_key}_mean"
            sd_col   = f"{lbl}_{base_key}_sd"
            if mean_col not in df.columns or df[mean_col].isna().all():
                continue
            y    = df[mean_col]
            yerr = df[sd_col].fillna(0.0) if sd_col in df.columns else 0.0
            ax.errorbar(
                x, y, yerr=yerr,
                fmt=f"{_DUAL_MARKERS[lbl]}-",
                color=_DUAL_COLORS_HEX[lbl],
                ecolor=_DUAL_COLORS_HEX[lbl],
                elinewidth=1, capsize=3, alpha=0.85,
                label=f"{lbl} mean ± SD",
            )
            plotted = True
        return plotted

    # Plot 1 — Diameter
    fig, ax = plt.subplots(figsize=(11, 5))
    if _dual_errbar(ax, "equiv_diameter_mm"):
        ax.set_xlabel(x_label)
        ax.set_ylabel("Equivalent Diameter (mm)")
        ax.set_title("Binned Drop Diameter — Control vs Sample")
        ax.legend(framealpha=0.8)
        _save_fig(fig, output_dir / "diameter_binned_dual.png", show)
    else:
        plt.close(fig)

    # Plot 2 — Volume
    fig, ax = plt.subplots(figsize=(11, 5))
    if _dual_errbar(ax, "volume_mm3"):
        ax.set_xlabel(x_label)
        ax.set_ylabel("Volume (mm³)")
        ax.set_title("Binned Drop Volume — Control vs Sample")
        ax.legend(framealpha=0.8)
        _save_fig(fig, output_dir / "volume_binned_dual.png", show)
    else:
        plt.close(fig)

    # Plot 3 — Eccentricity
    fig, ax = plt.subplots(figsize=(11, 5))
    if _dual_errbar(ax, "eccentricity"):
        ax.set_xlabel(x_label)
        ax.set_ylabel("Eccentricity")
        ax.set_title("Binned Drop Shape — Control vs Sample")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(framealpha=0.8)
        _save_fig(fig, output_dir / "eccentricity_binned_dual.png", show)
    else:
        plt.close(fig)

    # Plot 4 — Radius squared
    fig, ax = plt.subplots(figsize=(11, 5))
    if _dual_errbar(ax, "radius_eq_mm2"):
        ax.set_xlabel(x_label)
        ax.set_ylabel("$r_{eq}^2$ (mm²)")
        ax.set_title("Binned Radius Squared — Control vs Sample")
        ax.legend(framealpha=0.8)
        _save_fig(fig, output_dir / "radius_squared_binned_dual.png", show)
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Legacy single-drop plots (kept for backward compatibility with analyze_image.py)
# ---------------------------------------------------------------------------

def plot_timeseries(
    csv_path: str | Path,
    output_dir: str | Path | None = None,
    show: bool = False,
) -> None:
    """Generate time-series plots from a legacy single-drop CSV results file.

    .. note::
        This function targets the old, non-prefixed column schema produced
        by ``analyze_image.py``.  New code should call
        :func:`plot_dual_timeseries` instead.

    Creates plots for:
        1. Equivalent diameter vs. time
        2. Volume vs. time (if calibrated)
        3. Eccentricity vs. time
        4. Evaporation rate vs. time (if available)
        5. Radius squared vs. time (if available)

    Args:
        csv_path: Path to the CSV file with detection results.
        output_dir: Directory to save plot images.
        show: Whether to display plots interactively.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.error("CSV file not found: %s", csv_path)
        return

    df = pd.read_csv(csv_path)
    output_dir = Path(output_dir) if output_dir else csv_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x, x_label = _get_time_axis(df)
    plt.rcParams.update(_RCPARAMS)

    def _valid_invalid(col: str):
        if "tracking_valid" in df.columns:
            v = df["tracking_valid"] == True
            return v, ~v
        return pd.Series([True] * len(df)), pd.Series([False] * len(df))

    # Plot 1: Diameter
    fig, ax = plt.subplots(figsize=(10, 5))
    y_col = "equiv_diameter_mm" if "equiv_diameter_mm" in df.columns and df["equiv_diameter_mm"].notna().any() else "equiv_diameter_px"
    y_label = "Equivalent Diameter (mm)" if y_col == "equiv_diameter_mm" else "Equivalent Diameter (px)"
    valid, invalid = _valid_invalid(y_col)
    if valid.any():
        ax.plot(x[valid], df.loc[valid, y_col], "-", color="#2196F3", alpha=0.5)
        ax.scatter(x[valid], df.loc[valid, y_col], color="#2196F3", label="Valid", s=30, edgecolors="k", alpha=0.8)
    if invalid.any():
        ax.scatter(x[invalid], df.loc[invalid, y_col], color="#F44336", marker="x", label="Rejected", s=40, alpha=0.8)
    ax.set_ylabel(y_label); ax.set_xlabel(x_label); ax.set_title("Bubble Diameter Evolution"); ax.legend()
    _save_fig(fig, output_dir / "diameter_vs_time.png", show)

    # Plot 2: Volume
    if "volume_mm3" in df.columns and df["volume_mm3"].notna().any():
        fig, ax = plt.subplots(figsize=(10, 5))
        valid, invalid = _valid_invalid("volume_mm3")
        if valid.any():
            ax.plot(x[valid], df.loc[valid, "volume_mm3"], "-", color="#4CAF50", alpha=0.5)
            ax.scatter(x[valid], df.loc[valid, "volume_mm3"], color="#4CAF50", marker="s", label="Valid", s=30, edgecolors="k", alpha=0.8)
        if invalid.any():
            ax.scatter(x[invalid], df.loc[invalid, "volume_mm3"], color="#F44336", marker="x", label="Rejected", s=40, alpha=0.8)
        ax.set_xlabel(x_label); ax.set_ylabel("Volume (mm³)"); ax.set_title("Bubble Volume Evolution"); ax.legend()
        _save_fig(fig, output_dir / "volume_vs_time.png", show)

    # Plot 3: Eccentricity
    if "eccentricity" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        valid, invalid = _valid_invalid("eccentricity")
        if valid.any():
            ax.plot(x[valid], df.loc[valid, "eccentricity"], "-", color="#FF9800", alpha=0.5)
            ax.scatter(x[valid], df.loc[valid, "eccentricity"], color="#FF9800", marker="^", label="Valid", s=30, edgecolors="k", alpha=0.8)
        if invalid.any():
            ax.scatter(x[invalid], df.loc[invalid, "eccentricity"], color="#F44336", marker="x", label="Rejected", s=40, alpha=0.8)
        ax.set_xlabel(x_label); ax.set_ylabel("Eccentricity"); ax.set_title("Bubble Shape Evolution"); ax.set_ylim(-0.05, 1.05); ax.legend()
        _save_fig(fig, output_dir / "eccentricity_vs_time.png", show)

    # Plot 4: Evaporation rate
    if "evap_rate_mm3_s" in df.columns and df["evap_rate_mm3_s"].notna().any():
        fig, ax = plt.subplots(figsize=(10, 5))
        valid, _ = _valid_invalid("evap_rate_mm3_s")
        mask = valid & df["evap_rate_mm3_s"].notna()
        if mask.any():
            ax.plot(x[mask], df.loc[mask, "evap_rate_mm3_s"], "-", color="#F44336", alpha=0.5)
            ax.scatter(x[mask], df.loc[mask, "evap_rate_mm3_s"], color="#F44336", marker="d", label="dV/dt", s=30, edgecolors="k", alpha=0.8)
        ax.set_xlabel(x_label); ax.set_ylabel("Evaporation Rate (mm³/s)"); ax.set_title("Instantaneous Evaporation Rate"); ax.legend()
        _save_fig(fig, output_dir / "evaporation_rate.png", show)

    # Plot 5: Radius squared
    if "radius_eq_mm2" in df.columns and df["radius_eq_mm2"].notna().any():
        fig, ax = plt.subplots(figsize=(10, 5))
        valid, invalid = _valid_invalid("radius_eq_mm2")
        if valid.any():
            ax.plot(x[valid], df.loc[valid, "radius_eq_mm2"], "-", color="#2196F3", alpha=0.5)
            ax.scatter(x[valid], df.loc[valid, "radius_eq_mm2"], color="#2196F3", label="Valid", s=30, edgecolors="k", alpha=0.8)
        if invalid.any():
            ax.scatter(x[invalid], df.loc[invalid, "radius_eq_mm2"], color="#F44336", marker="x", label="Rejected", s=40, alpha=0.8)
        ax.set_xlabel(x_label); ax.set_ylabel("$r_{eq}^2$ (mm²)"); ax.set_title("Bubble Radius Squared Evolution"); ax.legend()
        _save_fig(fig, output_dir / "radius_squared_vs_time.png", show)


def plot_binned_timeseries(
    binned_csv_path: str | Path,
    output_dir: str | Path | None = None,
    show: bool = False,
) -> None:
    """Generate time-series plots with error bars from legacy single-drop binned results.

    .. note::
        Targets the old column schema (no label prefix).
        New code should call :func:`plot_binned_dual_timeseries`.

    Args:
        binned_csv_path: Path to the binned CSV file.
        output_dir: Output directory.
        show: Whether to display plots interactively.
    """
    binned_csv_path = Path(binned_csv_path)
    if not binned_csv_path.exists():
        logger.error("Binned CSV file not found: %s", binned_csv_path)
        return

    df = pd.read_csv(binned_csv_path)
    output_dir = Path(output_dir) if output_dir else binned_csv_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x = df["time_mean_s"]
    x_label = "Time (s)"
    plt.rcParams.update(_RCPARAMS)

    plots = [
        ("equivalent_diameter_mm", "Equivalent Diameter (mm)", "diameter_binned_vs_time.png", "#2196F3", "o-"),
        ("volume_mm3", "Volume (mm³)", "volume_binned_vs_time.png", "#4CAF50", "s-"),
        ("eccentricity", "Eccentricity", "eccentricity_binned_vs_time.png", "#FF9800", "^-"),
        ("radius_eq_mm2", "$r_{eq}^2$ (mm²)", "radius_squared_binned_vs_time.png", "#9C27B0", "o-"),
    ]
    for base, ylabel, fname, color, fmt in plots:
        mean_col = f"{base}_mean"
        sd_col   = f"{base}_sd"
        if mean_col not in df.columns or df[mean_col].isna().all():
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        yerr = df[sd_col].fillna(0.0) if sd_col in df.columns else 0.0
        ax.errorbar(x, df[mean_col], yerr=yerr, fmt=fmt, color=color,
                    ecolor=color, elinewidth=1, capsize=3, alpha=0.85,
                    label="Mean ± SD")
        ax.set_xlabel(x_label); ax.set_ylabel(ylabel)
        ax.set_title(f"Binned {ylabel.split('(')[0].strip()} Evolution")
        ax.legend()
        _save_fig(fig, output_dir / fname, show)
