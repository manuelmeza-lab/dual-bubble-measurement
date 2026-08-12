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
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Importaciones del paquete modular BubbleCV
from bubble_cv.calibration import calibrate
from bubble_cv.detection import BubbleDetection, detect_bubbles
from bubble_cv.io_utils import frame_iterator, load_image, save_csv
from bubble_cv.visualization import (
    plot_dual_timeseries,
    plot_binned_dual_timeseries,
    save_annotated_frame_dual,
)

# Configuración del registrador de logs del sistema
logger = logging.getLogger("bubble_cv")

# Etiquetas de las dos gotas (orden espacial: izq → der)
LABELS: tuple[str, str] = ("control", "sample")


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
        )

        # Accumulate bodyellipse audit (always, even when detection fails)
        for _side in LABELS:
            _ad = dict(dual["_audit"][_side])
            _ad["frame_id"] = frame_num
            _audit_records.append(_ad)

        if dual["control"] is None or dual["sample"] is None:
            logger.debug("Frame %d: dual detection failed.", frame_num)
            failed += 1
            continue

        ctrl_det: BubbleDetection = dual["control"]
        samp_det: BubbleDetection = dual["sample"]

        # QC físico independiente para cada gota
        validate_detection_physics(ctrl_det, max_eccentricity=args.max_eccentricity)
        validate_detection_physics(samp_det, max_eccentricity=args.max_eccentricity)

        # Construir fila pareada con columnas prefijadas
        row: dict = {
            "frame_id": frame_num,
            "timestamp_s": round(timestamp_s, 4),
            **_prefixed_dict(ctrl_det, "control"),
            **_prefixed_dict(samp_det, "sample"),
        }
        results.append(row)
        processed += 1

        if processed % 100 == 0:
            logger.info(
                "  Processed %d frames (frame #%d, t=%.1fs)...",
                processed, frame_num, timestamp_s,
            )

        # Visualización por fotograma (ambas gotas anotadas)
        if args.visualize:
            vis_path = Path(args.vis_dir) / f"frame_{frame_num:06d}.png"
            save_annotated_frame_dual(frame, ctrl_det, samp_det, vis_path)

    # ── Diagnóstico bodyellipse ──────────────────────────────────────
    _print_bodyellipse_audit(_audit_records)

    # ── Paso 4: Finalizar si no hubo detecciones ─────────────────────
    if not results:
        logger.error("No dual-drop detections in the entire video.")
        return 1

    # ── Paso 5: Construir DataFrame ───────────────────────────────────────
    df = pd.DataFrame(results)

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
    for lbl in LABELS:
        vol_col = f"{lbl}_volume_mm3"
        valid_col = f"{lbl}_tracking_valid"
        out_col = f"{lbl}_evap_rate_mm3_s"

        if vol_col in df.columns and df[vol_col].notna().any():
            df_valid = df[df[valid_col] == True].copy()
            if len(df_valid) > 1:
                rates = compute_evaporation_rate(
                    df_valid,
                    time_col="timestamp_s",
                    vol_col=vol_col,
                    out_col=out_col,
                )
                df[out_col] = rates.reindex(df.index)
            else:
                df[out_col] = None
        else:
            df[out_col] = None

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

        # Usar solo filas donde AMBAS gotas son válidas
        both_valid = (
            (df.get("control_tracking_valid", True) == True)
            & (df.get("sample_tracking_valid", True) == True)
        )
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
                        else:
                            row_b[f"{col}_mean"] = None
                            row_b[f"{col}_sd"] = 0.0
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

        total_frames = processed + failed
        summary_rows: list[dict] = []

        for lbl in LABELS:
            valid_col = f"{lbl}_tracking_valid"
            r2_col = f"{lbl}_radius_eq_mm2"

            df_lbl = df[df[valid_col] == True] if valid_col in df.columns else df
            valid_frames = len(df_lbl)
            rejected = failed + len(df[df.get(valid_col, pd.Series([True]*len(df))) == False])

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

            # Ajuste binned por gota
            if binned_df is not None and not binned_df.empty:
                b_r2_col = f"{lbl}_radius_eq_mm2_mean"
                df_bfit = binned_df[binned_df[b_r2_col].notna()] if b_r2_col in binned_df.columns else pd.DataFrame()
                n_bins = len(binned_df)
                if px_to_mm is not None and len(df_bfit) >= 2:
                    xb = df_bfit["time_mean_s"].values
                    yb = df_bfit[b_r2_col].values
                    b_slope, b_intercept, b_rsq = _linear_fit(xb, yb)
                    logger.info(
                        "  [%s binned] slope=%.6f  intercept=%.6f  R²=%.4f",
                        lbl, b_slope, b_intercept, b_rsq,
                    )

            summary_rows.append({
                "drop": lbl,
                "input_video": video_path.name,
                "total_frames": total_frames,
                "valid_frames": valid_frames,
                "rejected_frames": rejected,
                "fit_start_s": round(fit_start, 4) if fit_start is not None else None,
                "fit_end_s": round(fit_end, 4) if fit_end is not None else None,
                "slope_radius2_mm2_s": round(slope, 6) if slope is not None else None,
                "intercept_radius2_mm2": round(intercept, 6) if intercept is not None else None,
                "r_squared_fit": round(r_sq, 4) if r_sq is not None else None,
                "binned_slope_radius2_mm2_s": round(b_slope, 6) if b_slope is not None else None,
                "binned_intercept_radius2_mm2": round(b_intercept, 6) if b_intercept is not None else None,
                "binned_r_squared": round(b_rsq, 4) if b_rsq is not None else None,
                "bin_size_s": args.bin_size_s,
                "n_bins": n_bins,
            })

        pd.DataFrame(summary_rows).to_csv(args.summary_output, index=False)
        logger.info("Saved dual evaporation summary to %s", args.summary_output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
