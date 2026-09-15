"""
robust_fit.py — FIX 4: Estimador robusto Theil–Sen para BubbleCV Dual v0.2.0.

Implementa el ajuste lineal robusto Theil–Sen usado para calcular K a partir
de la mediana de ``radius_eq_mm2`` en cada bin temporal.

El estimador es ADITIVO: no reemplaza ni modifica el OLS existente.
Su única responsabilidad es proveer ``_theil_sen_fit`` como helper testeable.

Especificación algorítmica — CONGELADA (Fix 4):
------------------------------------------------

Pendientes pairwise:
    Para todos los pares i < j con dx = x[j] - x[i] != 0:
        slope_ij = (y[j] - y[i]) / dx

Pendiente Theil–Sen:
    slope = median(all valid pairwise slopes)

Intercepto joint (CONGELADO):
    intercept = median(y - slope * x)
    NO usar: median(y) - slope * median(x)

Coeficiente de determinación:
    y_pred  = slope * x + intercept
    SS_res  = sum((y - y_pred)^2)
    SS_tot  = sum((y - mean(y))^2)
    R²      = 1 - SS_res/SS_tot   si SS_tot != 0
    R²      = 0.0                  si SS_tot == 0
    (semántica idéntica a _linear_fit en analyze_video.py)

K:
    K = -slope
    NO usar abs(). Debe ser exactamente el negativo de la pendiente.

Filtrado de entrada:
    Eliminar solo filas donde x o y no son finitos (NaN, inf).
    Requiere al menos 2 puntos válidos y al menos un par con dx != 0.
    NO usar DUAL-QC para excluir puntos.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Helper público
# ---------------------------------------------------------------------------

def _theil_sen_fit(
    x: np.ndarray,
    y: np.ndarray,
) -> Optional[dict]:
    """Ajuste lineal robusto Theil–Sen.

    Calcula pendiente, intercepto, R² y K mediante el estimador Theil–Sen
    sobre todos los pares pairwise válidos.

    Args:
        x: Array de tiempos (``time_mean_s`` del bin).  Puede contener NaN/inf
           que serán filtrados.
        y: Array de valores a ajustar (``{side}_radius_eq_mm2_median`` del bin,
           ya redondeados a 4 decimales).  Misma longitud que ``x``.

    Returns:
        ``dict`` con claves::

            slope               float  — pendiente Theil–Sen
            intercept           float  — intercepto joint (median(y - slope*x))
            r_squared           float  — R²; 0.0 cuando SS_tot == 0
            n_pairwise_slopes   int    — número de pares pairwise usados
            K                   float  — -slope (negativo exacto, sin abs)

        ``None`` si no hay suficientes puntos válidos o no hay ninguna
        pendiente con dx != 0.

    Notes:
        El algoritmo está CONGELADO por la especificación de Fix 4.
        No modificar el orden de operaciones ni las condiciones de filtrado.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Filtrar filas donde x o y no son finitos
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]

    n = len(x)

    # Requiere al menos 2 puntos válidos
    if n < 2:
        return None

    # Calcular todas las pendientes pairwise para i < j con dx != 0
    slopes: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if dx != 0.0:
                slopes.append((y[j] - y[i]) / dx)

    # Requiere al menos una pendiente válida
    if not slopes:
        return None

    n_pairwise = len(slopes)
    slope = float(np.median(slopes))

    # Intercepto joint — CONGELADO: median(y - slope*x)
    intercept = float(np.median(y - slope * x))

    # R² respecto a la recta Theil–Sen
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    # Misma semántica que _linear_fit: R² = 0.0 cuando SS_tot == 0
    r_squared = (1.0 - ss_res / ss_tot) if ss_tot != 0.0 else 0.0

    # K = -slope exactamente (sin abs)
    K = -slope

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "n_pairwise_slopes": n_pairwise,
        "K": K,
    }
