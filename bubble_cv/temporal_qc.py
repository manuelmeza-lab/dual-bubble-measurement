"""
temporal_qc.py — FIX 3: Dual-QC Temporal como Diagnóstico.

Implementa verificaciones de consistencia temporal sobre la serie
``radius_eq_mm`` producida por el selector BODY (Fix 2).

El QC es EXCLUSIVAMENTE metadata/diagnóstico y NO participa en:
  - tracking_valid
  - geometry_quality_valid
  - paired validity
  - máscaras de binning
  - raw R² fit
  - dV/dt
  - cálculo de K

Arquitectura:
    detección BODY -> selector BODY (Fix 2) -> radius_eq_mm -> temporal QC

Las constantes están CONGELADAS y no son CLI-configurables.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes congeladas — FIX 3 (no modificar)
# ---------------------------------------------------------------------------

#: Error máximo de interpolación temporal permitido para veredicto PASS.
#: Derivado del audit histórico Gate 2A/2B sobre corrida3, corrida4, water_water.
TEMPORAL_QC_TOL_MM: float = 0.023748

#: Span máximo (next_t - prev_t) en segundos para que un check sea evaluable.
#: Superar este umbral clasifica el check como TEMPORAL_QC_UNAVAILABLE.
TEMPORAL_QC_MAX_BRIDGE_S: float = 20.0


# ---------------------------------------------------------------------------
# Función auxiliar interna
# ---------------------------------------------------------------------------

def _qc_check(
    t_i,
    r_i,
    prev_t,
    prev_r,
    next_t,
    next_r,
):
    """Ejecuta un check de QC temporal mediante interpolación lineal.

    El check es evaluable solo cuando:
      - Todos los valores (t_i, r_i, prev_t, prev_r, next_t, next_r) son finitos.
      - next_t > prev_t  (bridge span estrictamente positivo).
      - bridge_s = next_t - prev_t <= TEMPORAL_QC_MAX_BRIDGE_S.

    Fórmula:
        pred_mm  = prev_r + (t_i - prev_t) / bridge_s * (next_r - prev_r)
        error_mm = |r_i - pred_mm|
        ok       = error_mm <= TEMPORAL_QC_TOL_MM   (desigualdad <=, congelada)

    Args:
        t_i:    Timestamp del punto evaluado (segundos).
        r_i:    radius_eq_mm del punto evaluado.
        prev_t: Timestamp del vecino anterior (None o NaN si ausente).
        prev_r: radius_eq_mm del vecino anterior.
        next_t: Timestamp del vecino siguiente.
        next_r: radius_eq_mm del vecino siguiente.

    Returns:
        Tupla (pred_mm, error_mm, ok):
            pred_mm   — predicción interpolada, o None si unavailable.
            error_mm  — |r_i - pred_mm|, o None si unavailable.
            ok        — True/False segun la desigualdad <=, o None si unavailable.
    """
    # Todos los valores deben ser finitos
    for v in (t_i, r_i, prev_t, prev_r, next_t, next_r):
        if v is None:
            return None, None, None
        try:
            if not math.isfinite(float(v)):
                return None, None, None
        except (TypeError, ValueError):
            return None, None, None

    bridge_s = float(next_t) - float(prev_t)

    # Bridge span debe ser estrictamente positivo y dentro del límite
    if bridge_s <= 0.0 or bridge_s > TEMPORAL_QC_MAX_BRIDGE_S:
        return None, None, None

    # Interpolación lineal (lerp) sobre timestamps reales — NO promedio aritmético
    pred_mm = float(prev_r) + (float(t_i) - float(prev_t)) / bridge_s * (float(next_r) - float(prev_r))
    error_mm = abs(float(r_i) - pred_mm)
    ok = error_mm <= TEMPORAL_QC_TOL_MM  # desigualdad congelada: <=

    return pred_mm, error_mm, ok


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def add_temporal_qc(df):
    """Añade columnas de diagnóstico DUAL-QC temporal al DataFrame pareado.

    Opera sobre radius_eq_mm ya producido por el selector BODY (Fix 2).
    Paso de post-procesamiento puro: añade columnas nuevas, nunca modifica
    las existentes, no altera ninguna máscara analítica.

    Columnas añadidas para cada side en {'control', 'sample'}:

        {side}_temporal_qc_immediate_ok        bool | None
        {side}_temporal_qc_outer_ok            bool | None
        {side}_temporal_qc_dual_ok             bool
        {side}_temporal_qc_dual_available      bool
        {side}_temporal_qc_status              str
        {side}_temporal_qc_immediate_error_mm  float | None
        {side}_temporal_qc_outer_error_mm      float | None
        {side}_temporal_qc_immediate_pred_mm   float | None
        {side}_temporal_qc_outer_pred_mm       float | None

    Semántica:
      *_ok = True   -> check evaluable, error <= TEMPORAL_QC_TOL_MM
      *_ok = False  -> check evaluable, error > TEMPORAL_QC_TOL_MM
      *_ok = None   -> check no evaluable (vecino ausente, radio no finito,
                       bridge_s > MAX_BRIDGE_S, o next_t <= prev_t)

      dual_ok = True  SOLO cuando immediate_ok is True AND outer_ok is True
      dual_ok = False en CUALQUIER otro caso
      dual_available  = (immediate_ok is not None) AND (outer_ok is not None)

    Status:
      "PASS"                   -> dual_ok es True
      "FAIL"                   -> dual_available es True y dual_ok es False
      "TEMPORAL_QC_UNAVAILABLE"-> dual_available es False

    Args:
        df: DataFrame wide-format pareado de analyze_video.py.
            Debe contener 'timestamp_s' y '{side}_radius_eq_mm'.

    Returns:
        Nuevo DataFrame con todas las columnas originales mas las columnas
        temporal_qc_*. Mismo número y orden de filas que la entrada.
    """
    result = df.copy()
    n = len(df)

    if n == 0:
        for side in ("control", "sample"):
            for col in (
                f"{side}_temporal_qc_immediate_ok",
                f"{side}_temporal_qc_outer_ok",
                f"{side}_temporal_qc_dual_ok",
                f"{side}_temporal_qc_dual_available",
                f"{side}_temporal_qc_status",
                f"{side}_temporal_qc_immediate_error_mm",
                f"{side}_temporal_qc_outer_error_mm",
                f"{side}_temporal_qc_immediate_pred_mm",
                f"{side}_temporal_qc_outer_pred_mm",
            ):
                result[col] = pd.Series([], dtype=object)
        return result

    # Timestamps como array de float (NaN cuando no disponible)
    ts = pd.to_numeric(df["timestamp_s"], errors="coerce").to_numpy(dtype=float)

    for side in ("control", "sample"):
        rad_col = f"{side}_radius_eq_mm"

        if rad_col in df.columns:
            rs = pd.to_numeric(df[rad_col], errors="coerce").to_numpy(dtype=float)
        else:
            rs = np.full(n, float("nan"))

        # Arrays de salida — valores por defecto para caso unavailable
        imm_pred_v = [None] * n
        imm_err_v  = [None] * n
        imm_ok_v   = [None] * n  # None / True / False
        out_pred_v = [None] * n
        out_err_v  = [None] * n
        out_ok_v   = [None] * n
        dual_ok_v    = [False] * n
        dual_avail_v = [False] * n
        status_v     = ["TEMPORAL_QC_UNAVAILABLE"] * n

        for i in range(n):
            t_i = ts[i]
            r_i = rs[i]

            # --- QC INMEDIATO (k=1): vecinos i-1 / i+1 -------------------
            # Usar float("nan") como sentinel para índices fuera de rango;
            # evita el wrap-around negativo de numpy (ts[-1] = último elemento).
            prev_t_imm = ts[i - 1] if i >= 1     else float("nan")
            prev_r_imm = rs[i - 1] if i >= 1     else float("nan")
            next_t_imm = ts[i + 1] if i <= n - 2 else float("nan")
            next_r_imm = rs[i + 1] if i <= n - 2 else float("nan")

            i_pred, i_err, i_ok = _qc_check(
                t_i, r_i,
                prev_t_imm, prev_r_imm,
                next_t_imm, next_r_imm,
            )
            imm_pred_v[i] = i_pred
            imm_err_v[i]  = i_err
            imm_ok_v[i]   = i_ok

            # --- QC EXTERIOR (k=2): vecinos i-2 / i+2 --------------------
            prev_t_out = ts[i - 2] if i >= 2     else float("nan")
            prev_r_out = rs[i - 2] if i >= 2     else float("nan")
            next_t_out = ts[i + 2] if i <= n - 3 else float("nan")
            next_r_out = rs[i + 2] if i <= n - 3 else float("nan")

            o_pred, o_err, o_ok = _qc_check(
                t_i, r_i,
                prev_t_out, prev_r_out,
                next_t_out, next_r_out,
            )
            out_pred_v[i] = o_pred
            out_err_v[i]  = o_err
            out_ok_v[i]   = o_ok

            # --- DUAL ------------------------------------------------
            available = (i_ok is not None) and (o_ok is not None)
            dual_avail_v[i] = available

            if i_ok is True and o_ok is True:
                dual_ok_v[i] = True
                status_v[i]  = "PASS"
            elif available:
                dual_ok_v[i] = False
                status_v[i]  = "FAIL"
            else:
                dual_ok_v[i] = False
                status_v[i]  = "TEMPORAL_QC_UNAVAILABLE"

        # Asignar columnas
        result[f"{side}_temporal_qc_immediate_ok"]       = imm_ok_v
        result[f"{side}_temporal_qc_outer_ok"]           = out_ok_v
        result[f"{side}_temporal_qc_dual_ok"]            = dual_ok_v
        result[f"{side}_temporal_qc_dual_available"]     = dual_avail_v
        result[f"{side}_temporal_qc_status"]             = status_v
        result[f"{side}_temporal_qc_immediate_error_mm"] = imm_err_v
        result[f"{side}_temporal_qc_outer_error_mm"]     = out_err_v
        result[f"{side}_temporal_qc_immediate_pred_mm"]  = imm_pred_v
        result[f"{side}_temporal_qc_outer_pred_mm"]      = out_pred_v

    return result
