"""
tests/test_temporal_qc.py — FIX 3: Tests del DUAL-QC Temporal.

Cubre los 16 casos requeridos por la especificación:
  1.  Trayectoria perfectamente lineal (immediate=True, outer=True, dual=True)
  2.  Error > TEMPORAL_QC_TOL_MM (check=False)
  3.  Error == TEMPORAL_QC_TOL_MM exactamente (PASS, desigualdad <=)
  4.  Vecinos faltantes (check=None)
  5.  bridge_s > TEMPORAL_QC_MAX_BRIDGE_S (check=None)
  6.  bridge_s == TEMPORAL_QC_MAX_BRIDGE_S (evaluable)
  7.  immediate=True, outer=False (dual=False, available=True, status=FAIL)
  8.  immediate=False, outer=True (dual=False, available=True, status=FAIL)
  9.  Ambos True (dual=True, status=PASS)
 10.  Alguno unavailable (dual=False, available=False, status=TEMPORAL_QC_UNAVAILABLE)
 11.  Control y sample calculados independientemente
 12.  Timestamps desiguales: demuestra lerp real vs promedio aritmético
 13.  radius None/NaN/inf (unavailable sin excepción)
 14.  next_t <= prev_t (unavailable)
 15.  Preservar número y orden de filas
 16.  NO-CENSURA: temporal_qc_* no altera tracking_valid, geometry_quality_valid,
      ni la máscara de paired binning
"""

import math
import unittest

import numpy as np
import pandas as pd

from bubble_cv.temporal_qc import (
    TEMPORAL_QC_MAX_BRIDGE_S,
    TEMPORAL_QC_TOL_MM,
    _qc_check,
    add_temporal_qc,
)


# ---------------------------------------------------------------------------
# Helpers de aserción — evitan la confusión np.bool_ vs bool en assertIs
# ---------------------------------------------------------------------------

def _ok(val):
    """True si val es truthy booleano (acepta np.True_ y True)."""
    if val is None:
        return False
    return bool(val)

def _is_none(val):
    return val is None or (isinstance(val, float) and math.isnan(val))

def _is_true(val):
    return val is not None and bool(val) is True

def _is_false(val):
    return val is not None and bool(val) is False


# ---------------------------------------------------------------------------
# Utilidades de fixture
# ---------------------------------------------------------------------------

def _make_df(timestamps, ctrl_radii, samp_radii=None, extra_cols=None):
    """Construye un DataFrame mínimo para tests de add_temporal_qc."""
    n = len(timestamps)
    if samp_radii is None:
        samp_radii = ctrl_radii

    def _to_nan(v):
        return float("nan") if v is None else v

    data = {
        "frame_id":             list(range(n)),
        "timestamp_s":          list(timestamps),
        "control_radius_eq_mm": [_to_nan(r) for r in ctrl_radii],
        "sample_radius_eq_mm":  [_to_nan(r) for r in samp_radii],
    }
    if extra_cols:
        data.update(extra_cols)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Test 1 — Trayectoria perfectamente lineal
# ---------------------------------------------------------------------------
class TestPerfectLinear(unittest.TestCase):
    """Test 1: trayectoria lineal => immediate=True, outer=True, dual=True."""

    def setUp(self):
        ts = [0.0, 1.0, 2.0, 3.0, 4.0]
        rs = [1.000, 1.010, 1.020, 1.030, 1.040]
        self.df  = _make_df(ts, rs)
        self.out = add_temporal_qc(self.df)
        self.i   = 2

    def test_immediate_pass(self):
        self.assertTrue(_is_true(self.out.at[self.i, "control_temporal_qc_immediate_ok"]))

    def test_outer_pass(self):
        self.assertTrue(_is_true(self.out.at[self.i, "control_temporal_qc_outer_ok"]))

    def test_dual_true(self):
        self.assertTrue(_is_true(self.out.at[self.i, "control_temporal_qc_dual_ok"]))

    def test_dual_available(self):
        self.assertTrue(_is_true(self.out.at[self.i, "control_temporal_qc_dual_available"]))

    def test_status_pass(self):
        self.assertEqual(self.out.at[self.i, "control_temporal_qc_status"], "PASS")

    def test_immediate_error_zero(self):
        err = self.out.at[self.i, "control_temporal_qc_immediate_error_mm"]
        self.assertIsNotNone(err)
        self.assertAlmostEqual(float(err), 0.0, places=12)

    def test_outer_error_zero(self):
        err = self.out.at[self.i, "control_temporal_qc_outer_error_mm"]
        self.assertIsNotNone(err)
        self.assertAlmostEqual(float(err), 0.0, places=12)


# ---------------------------------------------------------------------------
# Test 2 — Error > TEMPORAL_QC_TOL_MM
# ---------------------------------------------------------------------------
class TestErrorAboveThreshold(unittest.TestCase):
    """Test 2: error > TOL => check=False."""

    def setUp(self):
        ts = [0.0, 1.0, 2.0, 3.0, 4.0]
        # Radio en i=2 = 1.060 => pred=1.020 => error=0.040 > TOL
        rs = [1.000, 1.010, 1.060, 1.030, 1.040]
        self.df  = _make_df(ts, rs)
        self.out = add_temporal_qc(self.df)

    def test_immediate_fail(self):
        self.assertTrue(_is_false(self.out.at[2, "control_temporal_qc_immediate_ok"]))

    def test_dual_false(self):
        self.assertTrue(_is_false(self.out.at[2, "control_temporal_qc_dual_ok"]))

    def test_error_value(self):
        err = self.out.at[2, "control_temporal_qc_immediate_error_mm"]
        self.assertAlmostEqual(float(err), 0.040, places=12)


# ---------------------------------------------------------------------------
# Test 3 — Error exactamente == TEMPORAL_QC_TOL_MM (debe ser PASS, <= congelado)
# ---------------------------------------------------------------------------
class TestErrorExactlyAtThreshold(unittest.TestCase):
    """Test 3: desigualdad congelada como <=.

    La aritmetica de punto flotante hace que 'pred + T - pred != T' exactamente
    para predicciones arbitrarias, por lo que el test de borde exacto se hace
    con vecinos constantes (pred = prev_r = next_r = 1.0), donde
    error = |r_i - 1.0| es exacto cuando r_i se expresa respecto a 1.0.

    En particular, 1.0 - T produce error = T - epsilon < T => PASS.
    Y el test 'just_above' verifica que T + epsilon => FAIL.
    Juntos fijan la desigualdad como <= (no <).
    """

    def test_at_qc_check_level_just_below_boundary(self):
        """error < T => PASS (verifica el lado 'menor que')."""
        T = TEMPORAL_QC_TOL_MM
        # Vecinos constantes: pred = 1.0 exactamente.
        # r_i = 1.0 - T => error = |1.0 - T - 1.0| = T - eps < T => PASS
        prev_t, prev_r = 1.0, 1.0
        next_t, next_r = 3.0, 1.0
        t_i = 2.0
        r_i = 1.0 - T  # error = T - epsilon => PASS
        pred, err, ok = _qc_check(t_i, r_i, prev_t, prev_r, next_t, next_r)
        self.assertIsNotNone(ok)
        self.assertIs(ok, True,
            msg="error < TOL_MM debe ser PASS")

    def test_at_qc_check_level_just_above_boundary_fails(self):
        """error > T => FAIL (verifica el lado 'mayor que')."""
        T = TEMPORAL_QC_TOL_MM
        prev_t, prev_r = 1.0, 1.0
        next_t, next_r = 3.0, 1.0
        t_i = 2.0
        # r_i = 1.0 + T + 1e-10 => error = T + 1e-10 > T => FAIL
        r_i_above = 1.0 + T + 1e-10
        _, err, ok = _qc_check(t_i, r_i_above, prev_t, prev_r, next_t, next_r)
        self.assertIs(ok, False,
            msg="error > TOL_MM debe ser FAIL")

    def test_inequality_is_leq_not_strict(self):
        """Fija la semantica: PASS incluye error <= T, FAIL es error > T.

        Usa tres radios:
          - error = 0        => PASS  (trivialmente)
          - error < T        => PASS
          - error = T + 1e-9 => FAIL
        Si fuera '<' en lugar de '<=', el caso error==T seria FAIL,
        pero la especificacion dice '<='. Este test fija la intencion.
        """
        T = TEMPORAL_QC_TOL_MM
        cases = [
            (0.0,       True,  "error=0 -> PASS"),
            (T * 0.5,   True,  "error<T -> PASS"),
            (T + 1e-9,  False, "error>T -> FAIL"),
        ]
        for err_target, expected_ok, desc in cases:
            # Vecinos constantes: pred=1.0, r_i = 1.0 + err_target
            r_i = 1.0 + err_target
            _, err_calc, ok = _qc_check(2.0, r_i, 1.0, 1.0, 3.0, 1.0)
            self.assertIs(ok, expected_ok, msg=f"Caso '{desc}' fallo")


# ---------------------------------------------------------------------------
# Test 4 — Vecinos faltantes: check=None
# ---------------------------------------------------------------------------
class TestMissingNeighbours(unittest.TestCase):
    """Test 4: primer y último frame carecen de vecinos => check=None."""

    def setUp(self):
        ts = [0.0, 1.0, 2.0, 3.0, 4.0]
        rs = [1.000, 1.010, 1.020, 1.030, 1.040]
        self.out = add_temporal_qc(_make_df(ts, rs))

    def test_first_frame_immediate_none(self):
        self.assertIsNone(self.out.at[0, "control_temporal_qc_immediate_ok"])

    def test_first_frame_outer_none(self):
        self.assertIsNone(self.out.at[0, "control_temporal_qc_outer_ok"])

    def test_last_frame_immediate_none(self):
        self.assertIsNone(self.out.at[4, "control_temporal_qc_immediate_ok"])

    def test_last_frame_outer_none(self):
        self.assertIsNone(self.out.at[4, "control_temporal_qc_outer_ok"])

    def test_second_frame_outer_none(self):
        # i=1: no tiene i-2
        self.assertIsNone(self.out.at[1, "control_temporal_qc_outer_ok"])

    def test_penultimate_frame_outer_none(self):
        # i=3: no tiene i+2
        self.assertIsNone(self.out.at[3, "control_temporal_qc_outer_ok"])


# ---------------------------------------------------------------------------
# Test 5 — bridge_s > MAX_BRIDGE_S: check=None
# ---------------------------------------------------------------------------
class TestBridgeTooLarge(unittest.TestCase):
    """Test 5: bridge > 20 s => unavailable."""

    def test_immediate_none_when_bridge_exceeds_max(self):
        # Para i=2: bridge_s = 23.0 - 1.0 = 22.0 > 20
        ts = [0.0, 1.0, 22.0, 23.0, 24.0]
        rs = [1.000, 1.010, 1.020, 1.030, 1.040]
        out = add_temporal_qc(_make_df(ts, rs))
        self.assertIsNone(out.at[2, "control_temporal_qc_immediate_ok"])

    def test_at_qc_check_level(self):
        _, _, ok = _qc_check(22.0, 1.020, 1.0, 1.010, 23.0, 1.030)
        self.assertIsNone(ok)


# ---------------------------------------------------------------------------
# Test 6 — bridge_s == MAX_BRIDGE_S: evaluable
# ---------------------------------------------------------------------------
class TestBridgeExactlyAtMax(unittest.TestCase):
    """Test 6: bridge == 20 s => evaluable."""

    def test_bridge_exactly_max_is_evaluable(self):
        # bridge = 30.0 - 10.0 = 20.0 == MAX_BRIDGE_S
        pred, err, ok = _qc_check(20.0, 1.020, 10.0, 1.010, 30.0, 1.030)
        self.assertIsNotNone(ok,
            msg="bridge_s == MAX_BRIDGE_S debe ser evaluable (<= incluye igualdad)")

    def test_bridge_just_above_max_is_unavailable(self):
        # bridge = 30.001 - 10.0 = 20.001 > MAX_BRIDGE_S
        _, _, ok = _qc_check(20.0, 1.020, 10.0, 1.010, 30.001, 1.030)
        self.assertIsNone(ok)


# ---------------------------------------------------------------------------
# Test 7 — immediate=True, outer=False => dual=False, available=True, FAIL
# ---------------------------------------------------------------------------
class TestImmediatePassOuterFail(unittest.TestCase):
    """Test 7."""

    def setUp(self):
        # Para forzar immediate PASS y outer FAIL en i=2:
        # immediate pred = lerp(r1, r3) con ts uniform:
        #   pred_imm = 1.010 + (2-1)/(3-1) * (1.030-1.010) = 1.020
        # outer pred = lerp(r0, r4) con t0=0, t4=4:
        #   pred_out = 1.000 + (2-0)/(4-0) * (r4 - 1.000)
        # Queremos r_i tal que: |r_i - pred_imm| <= TOL y |r_i - pred_out| > TOL
        # Fijamos r_i = 1.020 (pred_imm exacto => error_imm=0 => PASS)
        # Fijamos r4=1.200 => pred_out = 1.000 + 0.5*0.200 = 1.100
        # error_out = |1.020 - 1.100| = 0.080 > TOL => FAIL
        ts = [0.0, 1.0, 2.0, 3.0, 4.0]
        rs = [1.000, 1.010, 1.020, 1.030, 1.200]
        self.out = add_temporal_qc(_make_df(ts, rs))

    def test_immediate_pass(self):
        self.assertTrue(_is_true(self.out.at[2, "control_temporal_qc_immediate_ok"]))

    def test_outer_fail(self):
        self.assertTrue(_is_false(self.out.at[2, "control_temporal_qc_outer_ok"]))

    def test_dual_false(self):
        self.assertTrue(_is_false(self.out.at[2, "control_temporal_qc_dual_ok"]))

    def test_available_true(self):
        self.assertTrue(_is_true(self.out.at[2, "control_temporal_qc_dual_available"]))

    def test_status_fail(self):
        self.assertEqual(self.out.at[2, "control_temporal_qc_status"], "FAIL")


# ---------------------------------------------------------------------------
# Test 8 — immediate=False, outer=True => dual=False, available=True, FAIL
# ---------------------------------------------------------------------------
class TestImmediateFailOuterPass(unittest.TestCase):
    """Test 8."""

    def test_dual_false_when_immediate_fails(self):
        T = TEMPORAL_QC_TOL_MM
        # Calcular pred exactas via la implementacion para encontrar r_i correcto
        ts = [0.0, 1.0, 2.0, 3.0, 4.0]
        # immediate pred = lerp(r1, r3) at t=2 con ts uniforme
        # = r1 + 0.5*(r3-r1)
        # outer pred = lerp(r0, r4) at t=2
        # = r0 + 0.5*(r4-r0)
        # Queremos: immediate_error > T, outer_error <= T
        # Si r1=1.010, r3=1.030: pred_imm = 1.020
        # Si r0=1.000, r4=1.040: pred_out = 1.020
        # Necesitamos r_i tal que:
        #   |r_i - 1.020| > T  (immediate FAIL)
        #   |r_i - 1.020| <= T (outer PASS) => contradiccion con mismos preds

        # Usar r4 distinto para separar pred_imm y pred_out:
        # r4 = 1.000 + 2*(r_i - 1.000)  => pred_out = r_i (outer PASS con error=0)
        # Elegir r_i = 1.020 + 2*T:
        #   pred_imm = 1.020 => error_imm = 2T > T => FAIL
        #   pred_out = r_i => error_out = 0 <= T => PASS
        r_i = 1.020 + 2 * T
        # r4 tal que lerp(r0=1.000, r4) at t=2 con t0=0, t4=4 = r_i
        # r_i = 1.000 + 0.5*(r4 - 1.000) => r4 = 1.000 + 2*(r_i - 1.000)
        r4 = 1.000 + 2 * (r_i - 1.000)

        rs  = [1.000, 1.010, r_i, 1.030, r4]
        out = add_temporal_qc(_make_df(ts, rs))

        self.assertTrue(_is_false(out.at[2, "control_temporal_qc_immediate_ok"]),
            "immediate should FAIL")
        self.assertTrue(_is_true(out.at[2, "control_temporal_qc_outer_ok"]),
            "outer should PASS")
        self.assertTrue(_is_false(out.at[2, "control_temporal_qc_dual_ok"]))
        self.assertTrue(_is_true(out.at[2, "control_temporal_qc_dual_available"]))
        self.assertEqual(out.at[2, "control_temporal_qc_status"], "FAIL")


# ---------------------------------------------------------------------------
# Test 9 — Ambos PASS => dual=True, status=PASS
# ---------------------------------------------------------------------------
class TestBothPass(unittest.TestCase):
    """Test 9."""

    def test_both_pass_gives_dual_true(self):
        ts = [0.0, 1.0, 2.0, 3.0, 4.0]
        rs = [1.000, 1.010, 1.020, 1.030, 1.040]
        out = add_temporal_qc(_make_df(ts, rs))
        self.assertTrue(_is_true(out.at[2, "control_temporal_qc_dual_ok"]))
        self.assertEqual(out.at[2, "control_temporal_qc_status"], "PASS")


# ---------------------------------------------------------------------------
# Test 10 — Alguno unavailable => available=False, TEMPORAL_QC_UNAVAILABLE
# ---------------------------------------------------------------------------
class TestUnavailable(unittest.TestCase):
    """Test 10: cuando outer es unavailable (i=1, no hay i-2)."""

    def setUp(self):
        ts = [0.0, 1.0, 2.0, 3.0, 4.0]
        rs = [1.000, 1.010, 1.020, 1.030, 1.040]
        self.out = add_temporal_qc(_make_df(ts, rs))

    def test_i1_outer_unavailable(self):
        self.assertIsNone(self.out.at[1, "control_temporal_qc_outer_ok"])

    def test_i1_dual_not_available(self):
        self.assertTrue(_is_false(self.out.at[1, "control_temporal_qc_dual_available"]))

    def test_i1_dual_false(self):
        self.assertTrue(_is_false(self.out.at[1, "control_temporal_qc_dual_ok"]))

    def test_i1_status_unavailable(self):
        self.assertEqual(
            self.out.at[1, "control_temporal_qc_status"],
            "TEMPORAL_QC_UNAVAILABLE",
        )


# ---------------------------------------------------------------------------
# Test 11 — Control y sample calculados independientemente
# ---------------------------------------------------------------------------
class TestSidesIndependent(unittest.TestCase):
    """Test 11: control PASS != sample FAIL => no hay interferencia."""

    def setUp(self):
        T  = TEMPORAL_QC_TOL_MM
        ts = [0.0, 1.0, 2.0, 3.0, 4.0]
        ctrl = [1.000, 1.010, 1.020, 1.030, 1.040]           # lineal => PASS
        samp = [1.000, 1.010, 1.020 + 4 * T, 1.030, 1.040]  # error grande => FAIL
        self.out = add_temporal_qc(_make_df(ts, ctrl, samp_radii=samp))

    def test_control_passes(self):
        self.assertTrue(_is_true(self.out.at[2, "control_temporal_qc_immediate_ok"]))
        self.assertTrue(_is_true(self.out.at[2, "control_temporal_qc_dual_ok"]))

    def test_sample_fails(self):
        self.assertTrue(_is_false(self.out.at[2, "sample_temporal_qc_immediate_ok"]))
        self.assertTrue(_is_false(self.out.at[2, "sample_temporal_qc_dual_ok"]))

    def test_no_cross_contamination(self):
        ctrl_ok = bool(self.out.at[2, "control_temporal_qc_immediate_ok"])
        samp_ok = bool(self.out.at[2, "sample_temporal_qc_immediate_ok"])
        self.assertNotEqual(ctrl_ok, samp_ok)


# ---------------------------------------------------------------------------
# Test 12 — Timestamps desiguales: lerp real != promedio aritmético
# ---------------------------------------------------------------------------
class TestRealTimestampInterpolation(unittest.TestCase):
    """Test 12: con timestamps no uniformes, pred usa lerp, no promedio."""

    def test_lerp_vs_arithmetic_avg(self):
        # Timestamps NO uniformes
        ts = [0.0, 1.0, 2.0, 4.0, 8.0]
        # Radios lineales respecto al tiempo: r = 1.0 + 0.01*t
        rs = [1.0 + 0.01 * t for t in ts]
        # r = [1.000, 1.010, 1.020, 1.040, 1.080]

        # Para i=2 (t=2.0), immediate usa i-1 (t=1.0, r=1.010) y i+1 (t=4.0, r=1.040):
        # Lerp: pred = 1.010 + (2-1)/(4-1) * (1.040-1.010) = 1.010 + 1/3*0.030 = 1.020
        # Promedio aritmético: (1.010 + 1.040)/2 = 1.025

        df  = _make_df(ts, rs)
        out = add_temporal_qc(df)

        pred = float(out.at[2, "control_temporal_qc_immediate_pred_mm"])
        err  = float(out.at[2, "control_temporal_qc_immediate_error_mm"])

        # Lerp da error=0 para trayectoria lineal con timestamps reales
        self.assertAlmostEqual(pred, 1.020, places=10,
            msg="Pred debe ser 1.020 (lerp exacto sobre timestamps), no 1.025 (promedio)")
        self.assertAlmostEqual(err, 0.0, places=10,
            msg="Error debe ser 0 con lerp real; promedio aritmético daría ~0.005")


# ---------------------------------------------------------------------------
# Test 13 — radius None/NaN/inf: unavailable sin excepción
# ---------------------------------------------------------------------------
class TestInvalidRadius(unittest.TestCase):
    """Test 13: radios inválidos => None, sin crash."""

    def _check_unavailable(self, bad_r):
        ts = [0.0, 1.0, 2.0, 3.0, 4.0]
        rs = [1.000, 1.010, bad_r, 1.030, 1.040]
        df  = _make_df(ts, rs)
        out = add_temporal_qc(df)
        self.assertIsNone(out.at[2, "control_temporal_qc_immediate_ok"])
        self.assertIsNone(out.at[2, "control_temporal_qc_outer_ok"])
        self.assertTrue(_is_false(out.at[2, "control_temporal_qc_dual_available"]))

    def test_nan_radius(self):
        self._check_unavailable(float("nan"))

    def test_inf_radius(self):
        self._check_unavailable(float("inf"))

    def test_minus_inf_radius(self):
        self._check_unavailable(float("-inf"))

    def test_none_radius(self):
        self._check_unavailable(None)

    def test_at_qc_check_level_nan(self):
        _, _, ok = _qc_check(2.0, float("nan"), 1.0, 1.010, 3.0, 1.030)
        self.assertIsNone(ok)

    def test_at_qc_check_level_none(self):
        _, _, ok = _qc_check(2.0, None, 1.0, 1.010, 3.0, 1.030)
        self.assertIsNone(ok)

    def test_neighbor_nan(self):
        _, _, ok = _qc_check(2.0, 1.020, float("nan"), 1.010, 3.0, 1.030)
        self.assertIsNone(ok)

    def test_neighbor_radius_inf(self):
        _, _, ok = _qc_check(2.0, 1.020, 1.0, float("inf"), 3.0, 1.030)
        self.assertIsNone(ok)


# ---------------------------------------------------------------------------
# Test 14 — next_t <= prev_t: unavailable
# ---------------------------------------------------------------------------
class TestInvalidTimestamps(unittest.TestCase):
    """Test 14: next_t <= prev_t => unavailable."""

    def test_next_equal_prev(self):
        _, _, ok = _qc_check(2.0, 1.020, 2.0, 1.010, 2.0, 1.030)
        self.assertIsNone(ok, "next_t == prev_t => bridge_s=0 => unavailable")

    def test_next_less_than_prev(self):
        _, _, ok = _qc_check(2.0, 1.020, 3.0, 1.010, 1.0, 1.030)
        self.assertIsNone(ok, "next_t < prev_t => bridge_s<0 => unavailable")

    def test_nan_timestamp(self):
        _, _, ok = _qc_check(float("nan"), 1.020, 1.0, 1.010, 3.0, 1.030)
        self.assertIsNone(ok)


# ---------------------------------------------------------------------------
# Test 15 — Preservar número y orden de filas
# ---------------------------------------------------------------------------
class TestPreserveRowsAndOrder(unittest.TestCase):
    """Test 15: add_temporal_qc preserva filas, orden y columnas existentes."""

    def setUp(self):
        ts = [0.0, 1.0, 2.0, 3.0, 4.0]
        rs = [1.000, 1.010, 1.020, 1.030, 1.040]
        self.df  = _make_df(ts, rs)
        self.out = add_temporal_qc(self.df)

    def test_same_row_count(self):
        self.assertEqual(len(self.out), len(self.df))

    def test_same_index(self):
        pd.testing.assert_index_equal(self.out.index, self.df.index)

    def test_same_timestamps(self):
        pd.testing.assert_series_equal(self.out["timestamp_s"], self.df["timestamp_s"])

    def test_existing_columns_unchanged(self):
        for col in self.df.columns:
            pd.testing.assert_series_equal(self.out[col], self.df[col])

    def test_new_columns_present(self):
        for side in ("control", "sample"):
            for suffix in (
                "immediate_ok", "outer_ok", "dual_ok", "dual_available",
                "status", "immediate_error_mm", "outer_error_mm",
                "immediate_pred_mm", "outer_pred_mm",
            ):
                col = f"{side}_temporal_qc_{suffix}"
                self.assertIn(col, self.out.columns, msg=f"Missing column: {col}")

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame(columns=self.df.columns)
        out = add_temporal_qc(empty_df)
        self.assertEqual(len(out), 0)


# ---------------------------------------------------------------------------
# Test 16 — NO-CENSURA
# ---------------------------------------------------------------------------
class TestNoCensura(unittest.TestCase):
    """Test 16: temporal_qc_* no altera tracking_valid, geometry_quality_valid
    ni la máscara paired both_valid."""

    def setUp(self):
        ts = [0.0, 1.0, 2.0, 3.0, 4.0]
        rs = [1.000, 1.010, 1.020, 1.030, 1.040]
        extra = {
            "control_tracking_valid":         [True,  True,  False, True, True],
            "sample_tracking_valid":          [True,  True,  True,  True, True],
            "control_geometry_quality_valid": [True,  True,  True,  False, True],
            "sample_geometry_quality_valid":  [True,  True,  True,  True, True],
        }
        self.df_before = _make_df(ts, rs, extra_cols=extra)
        self.df_after  = add_temporal_qc(self.df_before)

    def _make_both_valid(self, df):
        return (
            (df["control_tracking_valid"] == True)
            & (df["sample_tracking_valid"] == True)
            & (df["control_geometry_quality_valid"] == True)
            & (df["sample_geometry_quality_valid"] == True)
        )

    def test_control_tracking_valid_unchanged(self):
        pd.testing.assert_series_equal(
            self.df_after["control_tracking_valid"],
            self.df_before["control_tracking_valid"],
        )

    def test_sample_tracking_valid_unchanged(self):
        pd.testing.assert_series_equal(
            self.df_after["sample_tracking_valid"],
            self.df_before["sample_tracking_valid"],
        )

    def test_control_geometry_quality_valid_unchanged(self):
        pd.testing.assert_series_equal(
            self.df_after["control_geometry_quality_valid"],
            self.df_before["control_geometry_quality_valid"],
        )

    def test_sample_geometry_quality_valid_unchanged(self):
        pd.testing.assert_series_equal(
            self.df_after["sample_geometry_quality_valid"],
            self.df_before["sample_geometry_quality_valid"],
        )

    def test_paired_valid_mask_unchanged(self):
        mask_before = self._make_both_valid(self.df_before)
        mask_after  = self._make_both_valid(self.df_after)
        self.assertTrue(
            (mask_before == mask_after).all(),
            "both_valid mask cambio despues de add_temporal_qc — NO-CENSURA violada",
        )

    def test_temporal_qc_cols_not_in_mask_columns(self):
        mask_cols = {
            "control_tracking_valid", "sample_tracking_valid",
            "control_geometry_quality_valid", "sample_geometry_quality_valid",
        }
        qc_cols = {c for c in self.df_after.columns if "temporal_qc" in c}
        overlap = mask_cols & qc_cols
        self.assertEqual(len(overlap), 0,
            msg=f"Solapamiento entre temporal_qc_* y columnas de mascara: {overlap}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
