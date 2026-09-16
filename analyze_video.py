#!/usr/bin/env python3
"""
analyze_video.py — Sistema de Procesamiento y Análisis de Evaporación Dual en Video.

Este script es el punto de entrada principal para analizar secuencias de video
adquiridas mediante microscopía DinoLite con DOS gotas colgantes simultáneas.
Su objetivo es detectar ambas gotas en cada fotograma, extraer sus propiedades
geométricas tridimensionales de forma independiente, aplicar filtros de calidad
física y calcular la tasa de evaporación lineal de manera comparativa.

El programa realiza las siguientes tareas principales:
1. Autocalibración espacial (px/mm) a partir de una imagen de referencia.
2. Lectura secuencial y eficiente de cuadros del video (con salto de cuadros ajustable).
3. Detección dual: 'control' (izq.) y 'sample' (der.) en cada fotograma.
4. Capa de Control de Calidad Físico (QC) para ambas gotas de forma independiente.
5. Exportación a un único CSV pareado con columnas prefijadas (control_*, sample_*).
6. Cálculo independiente de dV/dt para control y sample.
7. Ajuste lineal por mínimos cuadrados (r²_eq vs Tiempo) por separado para cada gota.
8. Generación de gráficos comparativos con ambas curvas superpuestas.

Uso:
    # Análisis básico a 30 FPS con calibración manual (100 px/mm):
    python analyze_video.py --input video.mov --calibration 100.0

    # Con salto de cuadros y ajuste lineal:
    python analyze_video.py --input video.mov --calibration 100.0 --skip 30 --r2-fit

    # Autocalibrando desde una imagen de referencia:
    python analyze_video.py --input video.mov --calibrate-from ref.png --fps 30 --r2-fit
"""

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Importaciones del paquete modular BubbleCV
from bubble_cv.calibration import calibrate
from bubble_cv.detection import BubbleDetection, detect_bubbles
from bubble_cv.io_utils import frame_iterator, load_image, save_csv
from bubble_cv.temporal_qc import add_temporal_qc
from bubble_cv.robust_fit import _theil_sen_fit
from bubble_cv.visualization import (
    plot_dual_timeseries,
    plot_binned_dual_timeseries,
    save_annotated_frame_dual,
)

# Configuración del registrador de logs del sistema
logger = logging.getLogger("bubble_cv")

# Etiquetas de las dos gotas (orden espacial: izq → der)
LABELS: tuple[str, str] = ("control", "sample")

# Umbral de calidad geométrica del ajuste bodyellipse.
# Basado exclusivamente en la distribución observada de residual_rmse:
# ajustes normales ≈ 0.035–0.05 ; detecciones incorrectas ≈ >0.14–0.20.
# El valor 0.08 se sitúa entre ambas poblaciones y fue elegido por
# criterio geométrico — NO optimizado contra K ni R².
BODYELLIPSE_MAX_RESIDUAL_RMSE: float = 0.08


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Configura y analiza los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="BubbleCV Dual — Analiza evaporación de dos gotas colgantes simultáneas.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # I/O
    parser.add_argument(
        "--input", "-i", required=True,
        help="Ruta al archivo de video de entrada (.mov, .mp4, .avi).",
    )
    parser.add_argument(
        "--output", "-o", default="results_dual.csv",
        help="Ruta para el CSV pareado con las mediciones de ambas gotas (default: results_dual.csv).",
    )

    # Parámetros de video
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="Tasa de fotogramas por segundo del video (default: 30.0).",
    )
    parser.add_argument(
        "--skip", "-s", type=int, default=1,
        help="Procesar 1 de cada N cuadros (default: 1).",
    )

    # Calibración
    cal_group = parser.add_mutually_exclusive_group()
    cal_group.add_argument(
        "--calibration", "-c", type=float, default=None,
        help="Factor de calibración manual en píxeles por milímetro (px/mm).",
    )
    cal_group.add_argument(
        "--calibrate-from", type=str, default=None,
        help="Ruta a una imagen de referencia para autocalibración.",
    )
    parser.add_argument(
        "--ref-diameter", type=float, default=4.0,
        help="Diámetro nominal de la esfera de referencia en mm (default: 4.0).",
    )

    # Detección
    parser.add_argument(
        "--clip-limit", type=float, default=3.0,
        help="Límite del contraste local CLAHE (default: 3.0).",
    )
    parser.add_argument(
        "--max-eccentricity", type=float, default=0.85,
        help="Límite de excentricidad máxima permitida (default: 0.85).",
    )
    parser.add_argument(
        "--control-body-start-y",
        type=int,
        default=None,
        help=(
            "Frontera física BODY fija para CONTROL, expresada como "
            "coordenada Y global del frame en píxeles. "
            "Si se omite, se conserva la detección automática RC3."
        ),
    )
    parser.add_argument(
        "--sample-body-start-y",
        type=int,
        default=None,
        help=(
            "Frontera física BODY fija para SAMPLE, expresada como "
            "coordenada Y global del frame en píxeles. "
            "Si se omite, se conserva la detección automática RC3."
        ),
    )

    # Postprocesamiento
    parser.add_argument(
        "--smooth", type=int, default=0,
        help="Tamaño de ventana para filtro de mediana móvil temporal (default: 0 = desactivado).",
    )
    parser.add_argument(
        "--r2-fit", action="store_true",
        help="Activa el ajuste lineal r²_eq vs tiempo para ambas gotas de forma independiente.",
    )
    parser.add_argument(
        "--summary-output", type=str, default="evaporation_summary_dual.csv",
        help="Ruta para el CSV con el resumen del ajuste lineal (default: evaporation_summary_dual.csv).",
    )
    parser.add_argument(
        "--bin-size-s", type=float, default=None,
        help="Tamaño del intervalo en segundos para análisis agrupado (binned).",
    )
    parser.add_argument(
        "--binned-output", type=str, default="results_dual_binned.csv",
        help="Ruta para el CSV del análisis binned (default: results_dual_binned.csv).",
    )

    # Visualización
    parser.add_argument(
        "--visualize", "-v", action="store_true",
        help="Guarda fotogramas anotados con las elipses de ambas gotas.",
    )
    parser.add_argument(
        "--vis-dir", type=str, default="annotated_frames",
        help="Carpeta destino para fotogramas anotados (default: annotated_frames/).",
    )
    parser.add_argument(
        "--plot", "-p", action="store_true",
        help="Genera gráficos comparativos de las series de tiempo de ambas gotas.",
    )

    # Logging
    parser.add_argument(
        "--verbose", action="store_true",
        help="Activa la salida de logs detallada (modo debug).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Geometry quality gate
# ---------------------------------------------------------------------------

def apply_geometry_quality_gate(
    detection: BubbleDetection,
    max_residual_rmse: float = BODYELLIPSE_MAX_RESIDUAL_RMSE,
) -> None:
    """Marca la bandera geometry_quality_valid en el objeto BubbleDetection.

    Criterio (todos deben cumplirse):
        * bodyellipse_used — verificado implicitamente: si residual_rmse es
          None, la bandera es False porque no hay ajuste disponible.
        * bodyellipse_residual_rmse es finito (no None, no NaN, no inf).
        * bodyellipse_residual_rmse ≤ max_residual_rmse.

    Modifica los campos in-place.  NO modifica las medidas geométricas
    (major_axis, minor_axis, equiv_diameter, volume, K) ni elimina la fila.

    Args:
        detection       : Objeto BubbleDetection ya procesado por el detector.
        max_residual_rmse: Umbral; por defecto BODYELLIPSE_MAX_RESIDUAL_RMSE.
    """
    rmse = detection.bodyellipse_residual_rmse

    if rmse is None or not math.isfinite(rmse):
        detection.geometry_quality_valid            = False
        detection.geometry_quality_rejection_reason = "bodyellipse_residual_rmse_missing"
        return

    if rmse > max_residual_rmse:
        detection.geometry_quality_valid            = False
        detection.geometry_quality_rejection_reason = "bodyellipse_residual_rmse"
        return

    detection.geometry_quality_valid            = True
    detection.geometry_quality_rejection_reason = ""


# ---------------------------------------------------------------------------
# Physics QC
# ---------------------------------------------------------------------------

def validate_detection_physics(
    detection: BubbleDetection,
    max_eccentricity: float = 0.85,
) -> None:
    """Ejecuta control de calidad físico (QC) en un objeto BubbleDetection.

    Modifica los campos ``tracking_valid`` y ``rejection_reason`` in-place.

    Reglas de rechazo:
        1. Excentricidad superior al umbral.
        2. Diámetro equivalente nulo o ≤ 0.
        3. Volumen nulo o ≤ 0.

    Args:
        detection: Resultado geométrico del detector.
        max_eccentricity: Límite de excentricidad máxima permitida.
    """
    reasons: list[str] = []

    if detection.eccentricity > max_eccentricity:
        reasons.append(
            f"eccentricity > {max_eccentricity} ({detection.eccentricity:.3f})"
        )
    if detection.equiv_diameter_mm is None or detection.equiv_diameter_mm <= 0:
        reasons.append("equiv_diameter_mm <= 0 or None")
    if detection.volume_mm3 is None or detection.volume_mm3 <= 0:
        reasons.append("volume_mm3 <= 0 or None")

    if reasons:
        detection.tracking_valid = False
        detection.rejection_reason = "; ".join(reasons)
    else:
        detection.tracking_valid = True
        detection.rejection_reason = ""


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------

def temporal_smooth(series: pd.Series, window: int) -> pd.Series:
    """Aplica un filtro de mediana móvil temporal sobre la serie de datos.

    Args:
        series: Serie de datos original.
        window: Ancho de la ventana móvil.

    Returns:
        Serie suavizada (sin modificar si window ≤ 1).
    """
    if window <= 1:
        return series
    return series.rolling(window, center=True, min_periods=1).median()


# ---------------------------------------------------------------------------
# Evaporation rate (per-drop)
# ---------------------------------------------------------------------------

def compute_evaporation_rate(
    df: pd.DataFrame,
    time_col: str = "timestamp_s",
    vol_col: str = "control_volume_mm3",
    out_col: str = "control_evap_rate_mm3_s",
) -> pd.Series:
    """Calcula la tasa de evaporación instantánea (derivada discreta dV/dt).

    Aplica diferencias finitas (dV / dt) sobre los datos de volumen de
    una sola gota.  Solo filas donde la gota es válida deben pasarse.

    Args:
        df: DataFrame con columnas de tiempo y volumen.
        time_col: Nombre de la columna de tiempo.
        vol_col: Nombre de la columna de volumen de la gota.
        out_col: Nombre de la serie de salida.

    Returns:
        pd.Series con la tasa de evaporación en mm³/s.
    """
    if vol_col not in df.columns or df[vol_col].isna().all():
        return pd.Series([None] * len(df), name=out_col, index=df.index)

    dt = df[time_col].diff()
    dv = df[vol_col].diff()
    rate = (dv / dt).replace([np.inf, -np.inf], np.nan)
    return rate.rename(out_col)


# ---------------------------------------------------------------------------
# CSV row builder (prefixed)
# ---------------------------------------------------------------------------

def _prefixed_dict(detection: BubbleDetection, label: str) -> dict:
    """Convierte un BubbleDetection a un dict con claves prefijadas por etiqueta.

    Args:
        detection: Resultado de detección de una gota.
        label: 'control' o 'sample'.

    Returns:
        Diccionario con claves del tipo ``control_volume_mm3``.
    """
    raw = detection.to_dict()
    # Excluir 'label' del dict porque ya está codificado en el prefijo
    return {f"{label}_{k}": v for k, v in raw.items() if k != "label"}


# ---------------------------------------------------------------------------
# Independent frame persistence — FIX 5
# ---------------------------------------------------------------------------

def _build_independent_frame_row(
    frame_num: int,
    timestamp_s: float,
    ctrl_det=None,
    samp_det=None,
) -> dict:
    """Construye una fila conservando cada detección de forma independiente.

    La fila existe aunque una o ambas gotas estén ausentes. Una detección
    presente aporta sus columnas prefijadas; una detección ausente no genera
    medidas geométricas ficticias.

    Los campos *_detected describen exclusivamente disponibilidad de la
    detección y NO sustituyen tracking_valid ni geometry_quality_valid.
    """
    row: dict = {
        "frame_id": frame_num,
        "timestamp_s": round(timestamp_s, 4),
        "control_detected": ctrl_det is not None,
        "sample_detected": samp_det is not None,
    }

    if ctrl_det is not None:
        row.update(_prefixed_dict(ctrl_det, "control"))

    if samp_det is not None:
        row.update(_prefixed_dict(samp_det, "sample"))

    return row


# ---------------------------------------------------------------------------
# Independent per-side binning — FIX 5
# ---------------------------------------------------------------------------

def _bin_side_independently(
    df: pd.DataFrame,
    side: str,
    bin_size_s: float,
) -> pd.DataFrame:
    """Agrupa una gota usando exclusivamente su propia validez analítica.

    Conserva la semántica estadística del binning histórico:

    * bin_id = floor(timestamp_s / bin_size_s);
    * time_mean_s = media de timestamps válidos del bin, redondeada a 4;
    * n_points = número de observaciones válidas del lado;
    * media y SD (ddof=1) de las métricas históricas;
    * SD = 0.0 cuando el bin contiene un único punto;
    * mediana de radius_eq_mm2 redondeada a 4 decimales antes del fit.

    No aplica temporal_qc_* y no utiliza información de la gota contralateral.
    No impone un mínimo nuevo de puntos por bin.
    """
    if bin_size_s <= 0.0:
        raise ValueError("bin_size_s must be > 0")

    mask_valid = _side_valid_mask(df, side)
    df_side = df[mask_valid].copy()

    columns = [
        "bin_id",
        "time_start_s",
        "time_end_s",
        "time_mean_s",
        "n_points",
    ]

    metrics = [
        "equiv_diameter_mm",
        "volume_mm3",
        "radius_eq_mm",
        "radius_eq_mm2",
        "eccentricity",
    ]

    for base in metrics:
        col = f"{side}_{base}"
        columns.extend([
            f"{col}_mean",
            f"{col}_sd",
        ])
        if base == "radius_eq_mm2":
            columns.append(f"{col}_median")

    if df_side.empty:
        return pd.DataFrame(columns=columns)

    df_side["bin_id"] = (
        df_side["timestamp_s"] // bin_size_s
    ).astype(int)

    binned_rows: list[dict] = []

    for bin_id, group in df_side.groupby("bin_id"):
        n = len(group)

        row_b: dict = {
            "bin_id": int(bin_id),
            "time_start_s": round(bin_id * bin_size_s, 4),
            "time_end_s": round((bin_id + 1) * bin_size_s, 4),
            "time_mean_s": round(group["timestamp_s"].mean(), 4),
            "n_points": n,
        }

        for base in metrics:
            col = f"{side}_{base}"

            if col in group.columns and group[col].notna().any():
                values = pd.to_numeric(
                    group[col],
                    errors="coerce",
                ).dropna()

                if not values.empty:
                    mean_value = values.mean()
                    sd_value = (
                        values.std(ddof=1)
                        if len(values) > 1
                        else 0.0
                    )

                    row_b[f"{col}_mean"] = round(
                        float(mean_value),
                        4,
                    )
                    row_b[f"{col}_sd"] = round(
                        float(sd_value)
                        if not pd.isna(sd_value)
                        else 0.0,
                        4,
                    )

                    if base == "radius_eq_mm2":
                        row_b[f"{col}_median"] = round(
                            float(values.median()),
                            4,
                        )
                    continue

            row_b[f"{col}_mean"] = None
            row_b[f"{col}_sd"] = 0.0

            if base == "radius_eq_mm2":
                row_b[f"{col}_median"] = None

        binned_rows.append(row_b)

    return pd.DataFrame(binned_rows, columns=columns)


# ---------------------------------------------------------------------------
# Independent per-side frame counts — FIX 5
# ---------------------------------------------------------------------------

def _side_frame_counts(df: pd.DataFrame, side: str) -> dict:
    """Resume detección y validez analítica para una gota independientemente.

    Distingue entre:

    * total_frames: todos los timestamps intentados;
    * detected_frames: frames donde el detector produjo esa gota;
    * valid_frames: detecciones que pasan la máscara analítica del lado;
    * missing_frames: frames sin detección para ese lado;
    * qc_rejected_frames: detectados pero no analíticamente válidos;
    * unusable_frames: todos los frames que no aportan al análisis del lado.

    La gota contralateral no participa en ninguna de estas cantidades.
    """
    total_frames = len(df)
    detected_col = f"{side}_detected"

    if detected_col in df.columns:
        detected_mask = df[detected_col].eq(True)
    else:
        detected_mask = pd.Series(False, index=df.index, dtype=bool)

    valid_mask = _side_valid_mask(df, side)

    detected_frames = int(detected_mask.sum())
    valid_frames = int(valid_mask.sum())

    missing_frames = total_frames - detected_frames
    qc_rejected_frames = int((detected_mask & ~valid_mask).sum())
    unusable_frames = total_frames - valid_frames

    return {
        "total_frames": total_frames,
        "detected_frames": detected_frames,
        "valid_frames": valid_frames,
        "missing_frames": missing_frames,
        "qc_rejected_frames": qc_rejected_frames,
        "unusable_frames": unusable_frames,
    }


# ---------------------------------------------------------------------------
# Independent per-side validity mask — FIX 5
# ---------------------------------------------------------------------------

def _side_valid_mask(df: pd.DataFrame, side: str) -> pd.Series:
    """Construye la máscara analítica válida de una gota de forma independiente.

    Una gota es analíticamente válida cuando, para ese mismo lado:

    * tracking_valid es True;
    * radius_eq_mm2 está disponible.

    ``geometry_quality_valid`` permanece disponible como diagnóstico de
    calidad del ajuste bodyellipse, pero no veta por sí sola una medición.

    La ausencia o invalidez de la gota contralateral NO participa en esta
    máscara. La validez pareada debe construirse explícitamente como la
    intersección de las máscaras individuales.

    Esta función NO usa temporal_qc_* y no modifica el DataFrame.
    """
    tracking_col = f"{side}_tracking_valid"
    radius_col = f"{side}_radius_eq_mm2"

    mask = pd.Series(True, index=df.index, dtype=bool)

    if tracking_col in df.columns:
        mask &= df[tracking_col].eq(True)
    else:
        mask &= False

    if radius_col in df.columns:
        radius = pd.to_numeric(df[radius_col], errors="coerce")
        mask &= radius.notna()
    else:
        mask &= False

    return mask


# ---------------------------------------------------------------------------
# Paired analytical validity — FIX 7
# ---------------------------------------------------------------------------

def _paired_valid_mask(df: pd.DataFrame) -> pd.Series:
    """Intersección de las validades analíticas de ambas gotas.

    La validez pareada no introduce un QC adicional. En particular,
    ``geometry_quality_valid`` permanece como diagnóstico y no actúa
    como veto analítico independiente.
    """
    return (
        _side_valid_mask(df, "control")
        & _side_valid_mask(df, "sample")
    )


# ---------------------------------------------------------------------------
# Independent per-side evaporation rate — FIX 7
# ---------------------------------------------------------------------------

def _compute_side_evaporation_rate(
    df: pd.DataFrame,
    side: str,
) -> pd.Series:
    """Calcula dV/dt usando la validez analítica central del propio lado.

    ``geometry_quality_valid`` permanece disponible como diagnóstico del
    ajuste bodyellipse, pero no veta por sí sola una medición.
    """
    vol_col = f"{side}_volume_mm3"
    out_col = f"{side}_evap_rate_mm3_s"

    out = pd.Series(
        np.nan,
        index=df.index,
        name=out_col,
        dtype=float,
    )

    if vol_col not in df.columns:
        return out

    volume = pd.to_numeric(df[vol_col], errors="coerce")
    mask = _side_valid_mask(df, side) & volume.notna()
    df_valid = df.loc[mask].copy()

    if len(df_valid) <= 1:
        return out

    rates = compute_evaporation_rate(
        df_valid,
        time_col="timestamp_s",
        vol_col=vol_col,
        out_col=out_col,
    )

    out.loc[rates.index] = rates
    return out


# ---------------------------------------------------------------------------
# Linear fit helper
# ---------------------------------------------------------------------------

def _linear_fit(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float]:
    """Ajuste lineal de mínimos cuadrados y = m*x + b.

    Args:
        x: Array de tiempos.
        y: Array de valores (r²_eq).

    Returns:
        Tupla (slope, intercept, r_squared).
    """
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0.0 else 0.0
    return float(slope), float(intercept), float(r_squared)


# ---------------------------------------------------------------------------
# Bodyellipse audit summary
# ---------------------------------------------------------------------------

def _print_bodyellipse_audit(records: list[dict]) -> None:
    """Print a per-side bodyellipse usage summary to the logger."""
    from collections import Counter

    logger.info("=" * 64)
    logger.info("BODYELLIPSE AUDIT")
    for side in ("control", "sample"):
        side_recs = [r for r in records if r["roi_label"] == side]
        total = len(side_recs)
        if total == 0:
            logger.info("%s:  no records collected.", side.upper())
            continue
        used     = sum(1 for r in side_recs if r["bodyellipse_used"])
        not_used = total - used
        pct_used = used / total * 100.0
        reasons  = Counter(
            r["bodyellipse_failure_reason"]
            for r in side_recs
            if not r["bodyellipse_used"]
        )
        logger.info("%s:", side.upper())
        logger.info("    total frames processed:   %d", total)
        logger.info("    bodyellipse used:          %d  (%.1f%%)", used, pct_used)
        logger.info("    fallback/non-bodyellipse:  %d  (%.1f%%)",
                    not_used, 100.0 - pct_used)
        if reasons:
            logger.info("    failure reasons:")
            for reason, count in reasons.most_common():
                logger.info("        %-38s %d", reason + ":", count)
    logger.info("=" * 64)


# ---------------------------------------------------------------------------
# Bodyellipse fit-quality audit summary
# ---------------------------------------------------------------------------

def _print_fit_quality_audit(df: pd.DataFrame) -> None:
    """Print per-side bodyellipse fit quality summary to the logger.

    After the FASE 3 refactor, ``best_cnt`` is an open physical arc.
    IoU, contour_area and area_ratio are intentionally ``None`` / NaN.
    This function reports only the metrics that remain geometrically valid:

    * ``fit_point_count``   — number of physical contour points sent to fitEllipse
    * ``residual_mean``     — mean algebraic distance to ellipse boundary
    * ``residual_rmse``     — RMS algebraic residual
    * ``residual_p95``      — 95th-percentile algebraic residual
    * ``ellipse_area_px2``  — π·a·b of the fitted ellipse (always valid)
    """
    logger.info("=" * 64)
    logger.info("BODYELLIPSE FIT QUALITY AUDIT")

    for side in ("control", "sample"):
        rmse_col  = f"{side}_bodyellipse_residual_rmse"
        mean_col  = f"{side}_bodyellipse_residual_mean"
        p95_col   = f"{side}_bodyellipse_residual_p95"
        pts_col   = f"{side}_bodyellipse_fit_point_count"
        earea_col = f"{side}_bodyellipse_ellipse_area_px2"
        maj_col   = f"{side}_major_axis_px"
        min_col   = f"{side}_minor_axis_px"

        # Gate on residual_rmse: if all NaN, no fit-quality data available
        if rmse_col not in df.columns or df[rmse_col].isna().all():
            logger.info("%s: no fit-quality data.", side.upper())
            continue

        # Work with rows that have a valid residual_rmse
        d = df[df[rmse_col].notna()].copy()
        n = len(d)

        logger.info("%s  (n=%d valid frames):", side.upper(), n)

        # ---- fit_point_count -----------------------------------------------
        if pts_col in d.columns and d[pts_col].notna().any():
            pc = d[pts_col].dropna()
            logger.info(
                "    fit point count:  mean=%.1f  median=%.1f  "
                "SD=%.1f  min=%d  max=%d",
                pc.mean(), pc.median(), pc.std(),
                int(pc.min()), int(pc.max()),
            )

        # ---- residual_mean -------------------------------------------------
        if mean_col in d.columns and d[mean_col].notna().any():
            rm = d[mean_col].dropna()
            logger.info(
                "    residual mean:    mean=%.4f  median=%.4f  SD=%.4f",
                rm.mean(), rm.median(), rm.std(),
            )

        # ---- residual_rmse -------------------------------------------------
        rr = d[rmse_col].dropna()
        logger.info(
            "    residual RMSE:    mean=%.4f  median=%.4f  SD=%.4f  max=%.4f",
            rr.mean(), rr.median(), rr.std(), rr.max(),
        )

        # ---- residual_p95 --------------------------------------------------
        if p95_col in d.columns and d[p95_col].notna().any():
            rp = d[p95_col].dropna()
            logger.info(
                "    residual P95:     mean=%.4f  median=%.4f  SD=%.4f  max=%.4f",
                rp.mean(), rp.median(), rp.std(), rp.max(),
            )

        # ---- ellipse_area_px2 (informative) --------------------------------
        if earea_col in d.columns and d[earea_col].notna().any():
            ea = d[earea_col].dropna()
            logger.info(
                "    ellipse area px²: mean=%.1f  median=%.1f  SD=%.1f",
                ea.mean(), ea.median(), ea.std(),
            )

        # ---- 5 frames with worst residual RMSE ----------------------------
        worst5 = d.nlargest(5, rmse_col)
        logger.info("    5 worst residual-RMSE frames:")
        for _, row in worst5.iterrows():
            logger.info(
                "        frame=%6d  t=%7.1fs  "
                "rmse=%.4f  p95=%.4f  pts=%s  major=%.1f  minor=%.1f",
                int(row.get("frame_id", -1)),
                float(row.get("timestamp_s", 0.0)),
                float(row[rmse_col]),
                float(row[p95_col])   if p95_col  in row and pd.notna(row[p95_col])  else float("nan"),
                str(int(row[pts_col])) if pts_col in row and pd.notna(row[pts_col]) else "?",
                float(row[maj_col])   if maj_col  in row and pd.notna(row[maj_col])  else float("nan"),
                float(row[min_col])   if min_col  in row and pd.notna(row[min_col])  else float("nan"),
            )

    logger.info("=" * 64)



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Función de entrada principal que orquesta el procesamiento dual del video."""
    args = parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Paso 1: Validar archivo de video ─────────────────────────────────
    video_path = Path(args.input)
    if not video_path.exists():
        logger.error("Video file not found: %s", video_path)
        return 1

    # ── Paso 2: Calibración física ────────────────────────────────────────
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
            clip_limit=args.clip_limit,
        )
        if px_to_mm is None:
            logger.error("Auto-calibration failed.")
            return 1
        logger.info("Calibration: %.2f px/mm", px_to_mm)

    if px_to_mm is None:
        logger.warning("No calibration provided. Results will be in pixel units only.")

    # ── Paso 3: Iteración y detección dual cuadro a cuadro ───────────────
    results: list[dict] = []
    processed = 0        # Fotogramas con detección exitosa de AMBAS gotas
    failed = 0           # Fotogramas donde la detección dual falló
    _audit_records: list[dict] = []   # Diagnóstico bodyellipse acumulado

    logger.info(
        "Processing video: %s (fps=%.1f, skip=%d)",
        video_path.name, args.fps, args.skip,
    )

    for frame_num, frame in frame_iterator(video_path, skip=args.skip):
        timestamp_s = frame_num / args.fps

        # Detección dual: retorna {'control': ..., 'sample': ...} o None
        dual = detect_bubbles(
            frame,
            px_to_mm=px_to_mm,
            clip_limit=args.clip_limit,
            control_body_start_y_global=args.control_body_start_y,
            sample_body_start_y_global=args.sample_body_start_y,
        )

        # Accumulate bodyellipse audit (always, even when detection fails)
        for _side in LABELS:
            _ad = dict(dual["_audit"][_side])
            _ad["frame_id"] = frame_num
            _audit_records.append(_ad)

        # FIX 5: persistencia independiente por lado.
        #
        # Una detección ausente ya NO elimina la detección válida de la
        # gota contralateral. ``processed`` conserva su semántica histórica:
        # número de frames donde AMBAS gotas fueron detectadas.
        ctrl_det = dual["control"]
        samp_det = dual["sample"]

        both_detected = ctrl_det is not None and samp_det is not None

        if both_detected:
            processed += 1
        else:
            failed += 1
            logger.debug(
                "Frame %d: one or both drop detections missing.",
                frame_num,
            )

        # QC físico y geométrico estrictamente independientes.
        if ctrl_det is not None:
            validate_detection_physics(
                ctrl_det,
                max_eccentricity=args.max_eccentricity,
            )
            apply_geometry_quality_gate(ctrl_det)

        if samp_det is not None:
            validate_detection_physics(
                samp_det,
                max_eccentricity=args.max_eccentricity,
            )
            apply_geometry_quality_gate(samp_det)

        # Siempre conservar el timestamp intentado.
        # Las medidas del lado ausente quedarán como NaN al construir
        # posteriormente el DataFrame.
        row = _build_independent_frame_row(
            frame_num=frame_num,
            timestamp_s=timestamp_s,
            ctrl_det=ctrl_det,
            samp_det=samp_det,
        )
        results.append(row)

        attempted = processed + failed
        if attempted % 100 == 0:
            logger.info(
                "  Attempted %d frames (frame #%d, t=%.1fs)...",
                attempted,
                frame_num,
                timestamp_s,
            )

        # La visualización histórica requiere ambas detecciones.
        # No inventar una elipse para el lado ausente.
        if args.visualize and both_detected:
            vis_path = Path(args.vis_dir) / f"frame_{frame_num:06d}.png"
            save_annotated_frame_dual(
                frame,
                ctrl_det,
                samp_det,
                vis_path,
            )

    # ── Diagnóstico bodyellipse ──────────────────────────────────────
    _print_bodyellipse_audit(_audit_records)

    # ── Paso 4: Finalizar si no hubo detecciones ─────────────────────
    if not results:
        logger.error("No dual-drop detections in the entire video.")
        return 1

    # ── Paso 5: Construir DataFrame ───────────────────────────────────────
    df = pd.DataFrame(results)

    # ── Paso 5b: DUAL-QC temporal (metadata/diagnóstico exclusivamente) ───
    # Opera sobre radius_eq_mm ya producido por el selector BODY (Fix 2).
    # NO modifica tracking_valid, geometry_quality_valid ni ninguna máscara
    # analítica (binning, R² fit, dV/dt, K).
    df = add_temporal_qc(df)

    # ── Calidad del ajuste bodyellipse ────────────────────────────────────
    _print_fit_quality_audit(df)

    # ── DETECTION QUALITY SUMMARY ─────────────────────────────────────────
    # FIX 5: contabilidad estrictamente independiente por gota.
    logger.info("=" * 64)
    logger.info("DETECTION QUALITY SUMMARY")
    logger.info("  RMSE threshold: %.4f", BODYELLIPSE_MAX_RESIDUAL_RMSE)

    for _lbl in LABELS:
        _counts = _side_frame_counts(df, _lbl)
        _gqv_col = f"{_lbl}_geometry_quality_valid"

        if _gqv_col in df.columns:
            _n_geom_ok = int((df[_gqv_col] == True).sum())
            _n_geom_rej = int((df[_gqv_col] == False).sum())
        else:
            _n_geom_ok = 0
            _n_geom_rej = 0

        _n_det = _counts["detected_frames"]
        _pct_geom_rej = (
            100.0 * _n_geom_rej / _n_det
            if _n_det > 0 else 0.0
        )

        logger.info(
            "  %s: attempted=%d  detections=%d  missing=%d  "
            "geometry_valid=%d  geometry_rejected=%d  "
            "geometry_rejected_pct=%.1f%%  analytical_valid=%d  "
            "qc_rejected=%d  unusable=%d",
            _lbl.upper(),
            _counts["total_frames"],
            _counts["detected_frames"],
            _counts["missing_frames"],
            _n_geom_ok,
            _n_geom_rej,
            _pct_geom_rej,
            _counts["valid_frames"],
            _counts["qc_rejected_frames"],
            _counts["unusable_frames"],
        )

    logger.info("=" * 64)

    # ── Paso 6: Suavizado temporal ────────────────────────────────────────

    if args.smooth > 1:
        logger.info("Applying temporal smoothing (window=%d)...", args.smooth)
        _base_px = ["equiv_diameter_px", "major_axis_px", "minor_axis_px", "eccentricity"]
        _base_mm = (
            ["equiv_diameter_mm", "major_axis_mm", "minor_axis_mm",
             "area_mm2", "surface_mm2", "volume_mm3"]
            if px_to_mm is not None else []
        )
        for lbl in LABELS:
            for base in _base_px + _base_mm:
                col = f"{lbl}_{base}"
                if col in df.columns:
                    df[col] = temporal_smooth(df[col], args.smooth)

    # ── Paso 7: Tasas de evaporación independientes (dV/dt) ───────────────
    # FIX 7: usar la misma máscara analítica central que binning y fits.
    # geometry_quality_valid se conserva como diagnóstico y no actúa aquí
    # como censor independiente.
    for lbl in LABELS:
        out_col = f"{lbl}_evap_rate_mm3_s"
        df[out_col] = _compute_side_evaporation_rate(df, lbl)

    # ── Paso 8: Exportar CSV pareado ──────────────────────────────────────
    df.to_csv(args.output, index=False)
    logger.info(
        "Done. Processed: %d frames, Failed: %d, Output: %s",
        processed, failed, args.output,
    )

    # ── Paso 9: Gráficos comparativos ─────────────────────────────────────
    if args.plot:
        logger.info("Generating dual comparison plots...")
        plot_dual_timeseries(args.output, show=False)

    # ── Paso 10: Análisis binned ──────────────────────────────────────────
    binned_df: pd.DataFrame | None = None

    if args.bin_size_s is not None and args.bin_size_s > 0.0:
        logger.info("Performing binned analysis (bin=%.2fs)...", args.bin_size_s)

        # FIX 7: la validez pareada es exclusivamente la intersección
        # de las validades analíticas individuales. El RMSE bodyellipse
        # permanece como diagnóstico y no introduce un veto adicional.
        both_valid = _paired_valid_mask(df)
        df_qc = df[both_valid].copy()

        if not df_qc.empty:
            df_qc["bin_id"] = (df_qc["timestamp_s"] // args.bin_size_s).astype(int)
            grouped = df_qc.groupby("bin_id")

            binned_rows: list[dict] = []
            for bin_id, group in grouped:
                n = len(group)
                row_b: dict = {
                    "bin_id": bin_id,
                    "time_start_s": round(bin_id * args.bin_size_s, 4),
                    "time_end_s": round((bin_id + 1) * args.bin_size_s, 4),
                    "time_mean_s": round(group["timestamp_s"].mean(), 4),
                    "n_points": n,
                }
                for lbl in LABELS:
                    for base in ["equiv_diameter_mm", "volume_mm3",
                                 "radius_eq_mm", "radius_eq_mm2", "eccentricity"]:
                        col = f"{lbl}_{base}"
                        if col in group.columns and group[col].notna().any():
                            m = group[col].mean()
                            s = group[col].std(ddof=1) if n > 1 else 0.0
                            row_b[f"{col}_mean"] = round(m, 4)
                            row_b[f"{col}_sd"] = round(float(s) if not pd.isna(s) else 0.0, 4)
                            # FIX 4: mediana de radius_eq_mm2 para Theil–Sen
                            # Almacenada con 4 decimales, que es el valor que entra al fit.
                            if base == "radius_eq_mm2":
                                row_b[f"{col}_median"] = round(group[col].median(), 4)
                        else:
                            row_b[f"{col}_mean"] = None
                            row_b[f"{col}_sd"] = 0.0
                            if base == "radius_eq_mm2":
                                row_b[f"{col}_median"] = None
                binned_rows.append(row_b)

            binned_df = pd.DataFrame(binned_rows)
            binned_df.to_csv(args.binned_output, index=False)
            logger.info("Saved binned results to %s", args.binned_output)

            logger.info("Generating binned dual comparison plots...")
            plot_binned_dual_timeseries(args.binned_output, show=False)
        else:
            logger.warning("No frames with both drops valid for binned analysis.")

    # ── Paso 11: Ajuste lineal R² independiente por gota ─────────────────
    if args.r2_fit:
        logger.info("Performing independent linear fit (r²_eq vs time) per drop...")

        total_frames = len(df)
        summary_rows: list[dict] = []

        for lbl in LABELS:
            r2_col = f"{lbl}_radius_eq_mm2"

            # FIX 5: validez y contabilidad independientes por gota.
            counts = _side_frame_counts(df, lbl)
            mask_valid = _side_valid_mask(df, lbl)
            df_lbl = df[mask_valid]

            valid_frames = counts["valid_frames"]
            rejected = counts["unusable_frames"]

            slope = intercept = r_sq = fit_start = fit_end = None
            b_slope = b_intercept = b_rsq = n_bins = None

            if px_to_mm is not None and r2_col in df_lbl.columns and valid_frames >= 2:
                x_fit = df_lbl["timestamp_s"].values
                y_fit = df_lbl[r2_col].dropna().values

                if len(y_fit) >= 2:
                    # Align lengths (drop NaN rows)
                    mask_nan = df_lbl[r2_col].notna()
                    x_fit = df_lbl.loc[mask_nan, "timestamp_s"].values
                    y_fit = df_lbl.loc[mask_nan, r2_col].values

                    fit_start = float(x_fit[0])
                    fit_end = float(x_fit[-1])
                    slope, intercept, r_sq = _linear_fit(x_fit, y_fit)
                    logger.info(
                        "  [%s] slope=%.6f  intercept=%.6f  R²=%.4f",
                        lbl, slope, intercept, r_sq,
                    )
            else:
                if px_to_mm is None:
                    logger.warning("[%s] No calibration — cannot perform R² fit.", lbl)
                else:
                    logger.warning("[%s] Fewer than 2 valid frames — cannot fit.", lbl)

            # FIX 5: binning analítico independiente por gota.
            #
            # results_dual_binned.csv permanece como producto PAREADO para
            # comparación simultánea. Los fits individuales, en cambio,
            # deben usar exclusivamente los puntos válidos de cada lado.
            side_binned_df = None

            if args.bin_size_s is not None and args.bin_size_s > 0.0:
                side_binned_df = _bin_side_independently(
                    df,
                    side=lbl,
                    bin_size_s=args.bin_size_s,
                )

            # Ajuste binned OLS por gota.
            if side_binned_df is not None and not side_binned_df.empty:
                b_r2_col = f"{lbl}_radius_eq_mm2_mean"

                if b_r2_col in side_binned_df.columns:
                    df_bfit = side_binned_df[
                        side_binned_df[b_r2_col].notna()
                    ]
                else:
                    df_bfit = pd.DataFrame()

                n_bins = len(side_binned_df)

                if px_to_mm is not None and len(df_bfit) >= 2:
                    xb = df_bfit["time_mean_s"].values
                    yb = df_bfit[b_r2_col].values

                    b_slope, b_intercept, b_rsq = _linear_fit(
                        xb,
                        yb,
                    )

                    logger.info(
                        "  [%s binned-independent] "
                        "slope=%.6f  intercept=%.6f  R²=%.4f  bins=%d",
                        lbl,
                        b_slope,
                        b_intercept,
                        b_rsq,
                        n_bins,
                    )

            # FIX 4 + FIX 5:
            # Theil–Sen conserva exactamente su algoritmo congelado,
            # pero recibe ahora los bins independientes de esta gota.
            #
            # La mediana radius_eq_mm2 ya está redondeada a 4 decimales
            # por _bin_side_independently().
            # NO usa DUAL-QC y no impone mínimo nuevo de puntos por bin.
            rob_slope = rob_intercept = rob_rsq = rob_K = rob_n = None

            if side_binned_df is not None and not side_binned_df.empty:
                rob_col = f"{lbl}_radius_eq_mm2_median"

                if rob_col in side_binned_df.columns and px_to_mm is not None:
                    df_rob = side_binned_df[
                        ["time_mean_s", rob_col]
                    ].copy()

                    df_rob = df_rob[
                        df_rob[rob_col].notna()
                    ]

                    if len(df_rob) >= 2:
                        xr = df_rob["time_mean_s"].values.astype(float)
                        yr = df_rob[rob_col].values.astype(float)

                        ts_result = _theil_sen_fit(xr, yr)

                        if ts_result is not None:
                            rob_slope = ts_result["slope"]
                            rob_intercept = ts_result["intercept"]
                            rob_rsq = ts_result["r_squared"]
                            rob_K = ts_result["K"]
                            rob_n = ts_result["n_pairwise_slopes"]

                            logger.info(
                                "  [%s robust-TS-independent] "
                                "K=%.9f  intercept=%.9f  "
                                "R²=%.9f  n_slopes=%d  bins=%d",
                                lbl,
                                rob_K,
                                rob_intercept,
                                rob_rsq,
                                rob_n,
                                len(side_binned_df),
                            )

            summary_rows.append({
                "drop": lbl,
                "input_video": video_path.name,
                "total_frames": total_frames,
                "detected_frames": counts["detected_frames"],
                "missing_frames": counts["missing_frames"],
                "valid_frames": valid_frames,
                "qc_rejected_frames": counts["qc_rejected_frames"],
                "unusable_frames": counts["unusable_frames"],
                # Campo histórico: conservar por compatibilidad.
                # Bajo FIX 5 equivale a total_frames - valid_frames.
                "rejected_frames": rejected,
                "fit_start_s": round(fit_start, 4) if fit_start is not None else None,
                "fit_end_s": round(fit_end, 4) if fit_end is not None else None,
                # --- columnas históricas OLS raw (conservar intactas) ---
                "slope_radius2_mm2_s": round(slope, 6) if slope is not None else None,
                "intercept_radius2_mm2": round(intercept, 6) if intercept is not None else None,
                "r_squared_fit": round(r_sq, 4) if r_sq is not None else None,
                # --- columnas históricas OLS binned (conservar intactas) ---
                "binned_slope_radius2_mm2_s": round(b_slope, 6) if b_slope is not None else None,
                "binned_intercept_radius2_mm2": round(b_intercept, 6) if b_intercept is not None else None,
                "binned_r_squared": round(b_rsq, 4) if b_rsq is not None else None,
                "bin_size_s": args.bin_size_s,
                "n_bins": n_bins,
                # --- FIX 4: campos robustos Theil–Sen (nuevos, inequívocos) ---
                "robust_binned_slope_radius2_mm2_s":   rob_slope,
                "robust_binned_intercept_radius2_mm2": rob_intercept,
                "robust_binned_r_squared":             rob_rsq,
                "robust_binned_K_mm2_s":               rob_K,
                "robust_binned_n_pairwise_slopes":     rob_n,
                "robust_binned_statistic":             "median",
                "robust_binned_fit_method":            "theil_sen_joint",
            })

        pd.DataFrame(summary_rows).to_csv(args.summary_output, index=False)
        logger.info("Saved dual evaporation summary to %s", args.summary_output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
