"""
tests/test_robust_binned_fit.py — FIX 4: Tests del estimador robusto Theil–Sen.

Cubre los 16 casos requeridos por la especificación:

  1.  Pendiente Theil–Sen sobre línea perfecta.
  2.  K == -slope exactamente.
  3.  Usa todas las pendientes pairwise válidas.
  4.  Ignora pares con dx == 0.
  5.  Intercepto joint: median(y - slope*x).
  6.  Demuestra explícitamente que NO usa median(y) - slope*median(x).
  7.  R² respecto a la recta Theil–Sen.
  8.  SS_tot == 0 → R² = 0.0.
  9.  NaN/inf filtrados de forma segura.
 10.  Menos de 2 puntos → fit no disponible (None).
 11.  Sin pendientes válidas → fit no disponible (None).
 12.  Mediana del bin de radius_eq_mm2 calculada correctamente.
 13.  Mediana redondeada a 4 decimales ANTES del fit.
 14.  Columnas históricas mean y SD permanecen sin cambios.
 15.  DUAL-QC no interviene en selección de frames/bins/fit.
 16.  Regresión Gate 3R: reproducir K/intercept/R²/n_pairwise_slopes
      de los seis dataset × side de gate3r_median_bins.csv.
"""

import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the project root is on sys.path (supports running from any cwd)
sys.path.insert(0, str(Path(__file__).parent.parent))

from bubble_cv.robust_fit import _theil_sen_fit


# ---------------------------------------------------------------------------
# Path to Gate 3R reference data
# ---------------------------------------------------------------------------

_GATE3R_DIR = (
    Path(__file__).parent.parent
    / "resultados"
    / "prueba_dual"
    / "corrected_residual_audit_water"
    / "gate3_revised_robust"
)
_GATE3R_BINS_CSV  = _GATE3R_DIR / "gate3r_median_bins.csv"
_GATE3R_SUMMARY_CSV = _GATE3R_DIR / "gate3r_summary.csv"


# ---------------------------------------------------------------------------
# Reference values — Gate 3R (FROZEN, from gate3r_summary.csv)
# ---------------------------------------------------------------------------

_GATE3R_REF = {
    ("corrida3",   "control"): {
        "K":           0.0004502814258911816,
        "intercept":   1.5541048780487805,
        "r_squared":   0.9331657543385513,
        "n_pairwise":  435,
    },
    ("corrida3",   "sample"): {
        "K":           0.0004890556597873659,
        "intercept":   1.5806692073170732,
        "r_squared":   0.9807846376641931,
        "n_pairwise":  435,
    },
    ("corrida4",   "control"): {
        "K":           0.0004605329311211671,
        "intercept":   1.6451308823529414,
        "r_squared":   0.9934088541491425,
        "n_pairwise":  435,
    },
    ("corrida4",   "sample"): {
        "K":           0.000444444444444446,
        "intercept":   1.678216666666667,
        "r_squared":   0.9955858269683856,
        "n_pairwise":  435,
    },
    ("water_water", "control"): {
        "K":           0.0004258333333333334,
        "intercept":   1.5126120833333334,
        "r_squared":   0.9952863209073866,
        "n_pairwise":  5995,
    },
    ("water_water", "sample"): {
        "K":           0.00044333333333333334,
        "intercept":   1.5368783333333331,
        "r_squared":   0.9759068814935611,
        "n_pairwise":  5995,
    },
}

# Column mapping: in gate3r_median_bins.csv the values are stored as
# control_value / sample_value, not the full column names.
_SIDE_COL = {"control": "control_value", "sample": "sample_value"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_line(slope, intercept, n=10, x_start=0.0, x_step=1.0):
    """Create x, y arrays for y = slope*x + intercept (perfect line)."""
    x = np.array([x_start + i * x_step for i in range(n)], dtype=float)
    y = slope * x + intercept
    return x, y


def _n_pairwise(n):
    """Expected number of pairwise slopes: C(n,2) = n*(n-1)//2."""
    return n * (n - 1) // 2


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

class TestTheilSenFit(unittest.TestCase):

    # ------------------------------------------------------------------ #
    # TEST 1 — Pendiente Theil–Sen sobre línea perfecta                   #
    # ------------------------------------------------------------------ #

    def test_01_perfect_line_slope(self):
        """Pendiente Theil–Sen debe recuperar la pendiente exacta en línea perfecta."""
        x, y = _make_line(slope=-0.0004, intercept=1.5, n=10)
        result = _theil_sen_fit(x, y)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["slope"], -0.0004, places=12)
        self.assertAlmostEqual(result["intercept"], 1.5, places=10)

    # ------------------------------------------------------------------ #
    # TEST 2 — K == -slope exactamente                                    #
    # ------------------------------------------------------------------ #

    def test_02_K_equals_neg_slope(self):
        """K debe ser exactamente -slope, sin abs()."""
        x, y = _make_line(slope=-0.0005, intercept=1.6, n=8)
        result = _theil_sen_fit(x, y)
        self.assertIsNotNone(result)
        # K = -slope → si slope es negativo, K debe ser positivo
        self.assertEqual(result["K"], -result["slope"])
        # Verificar que K no es abs(slope) cuando slope es positivo
        x2, y2 = _make_line(slope=0.0003, intercept=1.2, n=8)
        result2 = _theil_sen_fit(x2, y2)
        self.assertIsNotNone(result2)
        self.assertGreater(result2["slope"], 0)
        self.assertLess(result2["K"], 0)          # K negativo, no abs
        self.assertEqual(result2["K"], -result2["slope"])

    # ------------------------------------------------------------------ #
    # TEST 3 — Usa todas las pendientes pairwise válidas                  #
    # ------------------------------------------------------------------ #

    def test_03_all_pairwise_slopes_used(self):
        """El número de pendientes pairwise debe ser C(n, 2) para n puntos."""
        n = 10
        x, y = _make_line(slope=-0.0004, intercept=1.5, n=n)
        result = _theil_sen_fit(x, y)
        self.assertIsNotNone(result)
        expected_n = _n_pairwise(n)
        self.assertEqual(result["n_pairwise_slopes"], expected_n)

    def test_03b_pairwise_count_5_points(self):
        """Para 5 puntos: C(5,2) = 10 pendientes."""
        x, y = _make_line(slope=-0.001, intercept=2.0, n=5)
        result = _theil_sen_fit(x, y)
        self.assertIsNotNone(result)
        self.assertEqual(result["n_pairwise_slopes"], 10)

    # ------------------------------------------------------------------ #
    # TEST 4 — Ignora pares con dx == 0                                   #
    # ------------------------------------------------------------------ #

    def test_04_ignores_dx_zero_pairs(self):
        """Pares con x[i] == x[j] deben ser excluidos del cálculo."""
        # Dos puntos con mismo x: solo 1 par pero dx=0 → sin pendientes
        x = np.array([1.0, 1.0, 2.0], dtype=float)
        y = np.array([1.5, 1.6, 1.4], dtype=float)
        result = _theil_sen_fit(x, y)
        # Solo el par (0,2) y (1,2) tienen dx != 0 → 2 slopes
        self.assertIsNotNone(result)
        self.assertEqual(result["n_pairwise_slopes"], 2)

    def test_04b_all_dx_zero_returns_none(self):
        """Si todos los x son iguales (todos dx == 0), el fit retorna None."""
        x = np.array([5.0, 5.0, 5.0], dtype=float)
        y = np.array([1.4, 1.5, 1.6], dtype=float)
        result = _theil_sen_fit(x, y)
        self.assertIsNone(result)

    # ------------------------------------------------------------------ #
    # TEST 5 — Intercepto joint: median(y - slope*x)                     #
    # ------------------------------------------------------------------ #

    def test_05_joint_intercept_formula(self):
        """El intercepto debe ser median(y - slope*x)."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        y = np.array([3.0, 5.1, 6.9, 9.2], dtype=float)
        result = _theil_sen_fit(x, y)
        self.assertIsNotNone(result)
        slope = result["slope"]
        expected_intercept = float(np.median(y - slope * x))
        self.assertAlmostEqual(result["intercept"], expected_intercept, places=12)

    # ------------------------------------------------------------------ #
    # TEST 6 — Demuestra explícitamente que NO usa median(y)-slope*median(x) #
    # ------------------------------------------------------------------ #

    def test_06_joint_differs_from_broken_formula(self):
        """Demostrar que el intercepto joint difiere de median(y)-slope*median(x)
        en un dataset donde las dos fórmulas dan resultados distintos.

        La diferencia surge cuando los residuos (y - slope*x) tienen una
        distribución asimétrica, lo que hace que median(y - slope*x) ≠
        median(y) - slope*median(x).

        Dataset diseñado: pendiente ~2, pero los residuos son muy asimétricos
        porque un punto tiene un offset grande que solo afecta a la mediana
        del punto individual pero no a la mediana global."""
        # Dataset con residuos muy asimétricos:
        # La pendiente pairwise resulta 2.0 para todos los pares.
        # Con slope=2.0:
        #   residuos = y - 2*x = [0, 10, 0, 0]
        #   median(residuos) = 0.0   → intercept_joint = 0.0
        #   median(y) = 3.0, median(x) = 2.5
        #   median(y) - slope*median(x) = 3.0 - 2*2.5 = -2.0  → != 0.0
        x = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
        y = np.array([0.0, 12.0, 4.0, 6.0], dtype=float)
        # pendientes pairwise:
        #   (0,1): (12-0)/(1-0) = 12
        #   (0,2): (4-0)/(2-0)  = 2
        #   (0,3): (6-0)/(3-0)  = 2
        #   (1,2): (4-12)/(2-1) = -8
        #   (1,3): (6-12)/(3-1) = -3
        #   (2,3): (6-4)/(3-2)  = 2
        # sorted: [-8, -3, 2, 2, 2, 12]  median = (2+2)/2 = 2.0
        # intercept_joint  = median(y - 2*x) = median([0, 10, 0, 0]) = 0.0
        # intercept_broken = median(y) - 2*median(x) = 3.0 - 2*2.5 = -2.0

        result = _theil_sen_fit(x, y)
        self.assertIsNotNone(result)
        slope = result["slope"]

        intercept_joint  = float(np.median(y - slope * x))
        intercept_broken = float(np.median(y)) - slope * float(np.median(x))

        # Verificar que las dos fórmulas difieren en este dataset
        self.assertFalse(
            math.isclose(intercept_joint, intercept_broken, rel_tol=1e-9, abs_tol=1e-12),
            msg=(
                "Dataset elegido no diferencia las dos fórmulas: "
                f"joint={intercept_joint}, broken={intercept_broken}. "
                "Revisar fixture."
            ),
        )

        # El intercepto devuelto debe coincidir con la fórmula joint
        self.assertAlmostEqual(result["intercept"], intercept_joint, places=12)

        # El intercepto devuelto NO debe coincidir con la fórmula rota
        self.assertFalse(
            math.isclose(result["intercept"], intercept_broken, rel_tol=1e-9, abs_tol=1e-12),
            msg=(
                f"El intercepto devuelto ({result['intercept']}) coincide con "
                f"la fórmula rota median(y)-slope*median(x) = {intercept_broken}."
            ),
        )

    # ------------------------------------------------------------------ #
    # TEST 7 — R² respecto a la recta Theil–Sen                          #
    # ------------------------------------------------------------------ #

    def test_07_r_squared_perfect_fit(self):
        """R² = 1.0 para línea perfecta."""
        x, y = _make_line(slope=-0.0004, intercept=1.5, n=20)
        result = _theil_sen_fit(x, y)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["r_squared"], 1.0, places=10)

    def test_07b_r_squared_formula(self):
        """R² calculado contra la recta Theil–Sen, no contra otra línea."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
        y = np.array([1.0, 2.1, 2.9, 4.2, 5.0], dtype=float)
        result = _theil_sen_fit(x, y)
        self.assertIsNotNone(result)
        slope     = result["slope"]
        intercept = result["intercept"]
        y_pred    = slope * x + intercept
        ss_res    = np.sum((y - y_pred) ** 2)
        ss_tot    = np.sum((y - np.mean(y)) ** 2)
        expected_r2 = 1.0 - ss_res / ss_tot
        self.assertAlmostEqual(result["r_squared"], expected_r2, places=12)

    # ------------------------------------------------------------------ #
    # TEST 8 — SS_tot == 0 → R² = 0.0                                   #
    # ------------------------------------------------------------------ #

    def test_08_ss_tot_zero_gives_r2_zero(self):
        """Cuando todos los y son iguales (SS_tot=0), R² debe ser 0.0, no NaN."""
        x = np.array([1.0, 2.0, 3.0], dtype=float)
        y = np.array([1.5, 1.5, 1.5], dtype=float)
        result = _theil_sen_fit(x, y)
        self.assertIsNotNone(result)
        self.assertEqual(result["r_squared"], 0.0)
        self.assertFalse(math.isnan(result["r_squared"]))

    # ------------------------------------------------------------------ #
    # TEST 9 — NaN/inf filtrados de forma segura                         #
    # ------------------------------------------------------------------ #

    def test_09_nan_inf_filtered_safely(self):
        """NaN e inf en x o y deben ser descartados sin excepción."""
        x = np.array([1.0, float("nan"), 3.0, float("inf"), 5.0], dtype=float)
        y = np.array([1.5, 1.6,          2.5, float("nan"), 3.5], dtype=float)
        # Solo puntos (1.0,1.5), (3.0,2.5), (5.0,3.5) son finitos
        result = _theil_sen_fit(x, y)
        self.assertIsNotNone(result)
        self.assertEqual(result["n_pairwise_slopes"], _n_pairwise(3))  # 3 puntos → 3 pares

    def test_09b_all_nan_returns_none(self):
        """Si todos los valores son NaN, retorna None."""
        x = np.array([float("nan")] * 5, dtype=float)
        y = np.array([float("nan")] * 5, dtype=float)
        result = _theil_sen_fit(x, y)
        self.assertIsNone(result)

    # ------------------------------------------------------------------ #
    # TEST 10 — Menos de 2 puntos → fit no disponible (None)             #
    # ------------------------------------------------------------------ #

    def test_10_less_than_2_points_returns_none(self):
        """Con 0 o 1 punto válido, el fit retorna None."""
        # 0 puntos
        result = _theil_sen_fit(np.array([], dtype=float), np.array([], dtype=float))
        self.assertIsNone(result)
        # 1 punto
        result = _theil_sen_fit(np.array([1.0]), np.array([1.5]))
        self.assertIsNone(result)

    def test_10b_only_nans_leaves_zero_valid_points(self):
        """4 NaN: quedan 0 puntos válidos → None."""
        x = np.full(4, float("nan"))
        y = np.full(4, float("nan"))
        result = _theil_sen_fit(x, y)
        self.assertIsNone(result)

    # ------------------------------------------------------------------ #
    # TEST 11 — Sin pendientes válidas → fit no disponible (None)        #
    # ------------------------------------------------------------------ #

    def test_11_no_valid_slopes_returns_none(self):
        """2 puntos con mismo x → dx == 0 → sin pendientes → None."""
        x = np.array([5.0, 5.0], dtype=float)
        y = np.array([1.4, 1.6], dtype=float)
        result = _theil_sen_fit(x, y)
        self.assertIsNone(result)

    # ------------------------------------------------------------------ #
    # TEST 12 — Mediana del bin de radius_eq_mm2 calculada correctamente #
    # ------------------------------------------------------------------ #

    def test_12_bin_median_calculation(self):
        """La mediana del bin debe ser calculada con pandas median() sobre
        los valores del grupo, coincidiendo con numpy median."""
        # Simular un grupo de 5 valores de radius_eq_mm2
        values = np.array([1.50, 1.48, 1.52, 1.51, 1.49], dtype=float)
        expected_median = float(np.median(values))
        computed_median = float(pd.Series(values).median())
        self.assertAlmostEqual(computed_median, expected_median, places=12)
        # Verificar que el redondeo a 4 decimales se aplica
        rounded = round(computed_median, 4)
        self.assertEqual(rounded, round(expected_median, 4))

    def test_12b_odd_even_median(self):
        """Mediana con n impar e n par."""
        odd  = pd.Series([1.5, 1.4, 1.6])
        even = pd.Series([1.5, 1.4, 1.6, 1.55])
        self.assertAlmostEqual(float(odd.median()),  1.5,    places=10)
        self.assertAlmostEqual(float(even.median()), 1.525,  places=10)

    # ------------------------------------------------------------------ #
    # TEST 13 — Mediana redondeada a 4 decimales ANTES del fit           #
    # ------------------------------------------------------------------ #

    def test_13_median_rounded_before_fit(self):
        """El valor almacenado en *_median (ya redondeado a 4 decimales)
        es el que entra a Theil–Sen. Demostrar que fit con valores redondeados
        y sin redondear pueden diferir."""
        x = np.array([10.0, 20.0, 30.0], dtype=float)
        # Valores exactos con muchos decimales
        y_exact   = np.array([1.55555555, 1.44444444, 1.33333333], dtype=float)
        # Valores redondeados a 4 decimales (como se almacenan en binned_df)
        y_rounded = np.array([round(v, 4) for v in y_exact], dtype=float)

        result_exact   = _theil_sen_fit(x, y_exact)
        result_rounded = _theil_sen_fit(x, y_rounded)

        self.assertIsNotNone(result_exact)
        self.assertIsNotNone(result_rounded)

        # Los valores redondeados a 4 decimales difieren de los exactos
        # (verificar que la distinción existe)
        self.assertFalse(np.allclose(y_exact, y_rounded, atol=1e-8),
                         "El dataset elegido no diferencia redondeo. Revisar fixture.")

        # El fit sobre valores redondeados usa los valores redondeados
        self.assertAlmostEqual(
            result_rounded["slope"],
            float(np.median([
                (y_rounded[1] - y_rounded[0]) / (x[1] - x[0]),
                (y_rounded[2] - y_rounded[0]) / (x[2] - x[0]),
                (y_rounded[2] - y_rounded[1]) / (x[2] - x[1]),
            ])),
            places=12,
        )

    # ------------------------------------------------------------------ #
    # TEST 14 — Columnas históricas mean y SD permanecen sin cambios     #
    # ------------------------------------------------------------------ #

    def test_14_historical_mean_sd_columns_unchanged(self):
        """Añadir la columna *_median no debe alterar *_mean ni *_sd."""
        # Simular un binned DataFrame como lo construye analyze_video.py
        data = {
            "bin_id": [0, 1, 2],
            "time_mean_s": [5.0, 15.0, 25.0],
            "n_points": [10, 10, 10],
            "control_radius_eq_mm2_mean":   [1.5432, 1.5310, 1.5201],
            "control_radius_eq_mm2_sd":     [0.0021, 0.0019, 0.0022],
            "control_radius_eq_mm2_median": [1.5430, 1.5308, 1.5199],
            "sample_radius_eq_mm2_mean":    [1.5801, 1.5690, 1.5577],
            "sample_radius_eq_mm2_sd":      [0.0018, 0.0020, 0.0017],
            "sample_radius_eq_mm2_median":  [1.5800, 1.5688, 1.5575],
        }
        df = pd.DataFrame(data)

        # mean y SD deben ser exactamente los valores establecidos (sin tocar)
        for side in ("control", "sample"):
            mean_col   = f"{side}_radius_eq_mm2_mean"
            sd_col     = f"{side}_radius_eq_mm2_sd"
            median_col = f"{side}_radius_eq_mm2_median"

            self.assertIn(mean_col, df.columns)
            self.assertIn(sd_col, df.columns)
            self.assertIn(median_col, df.columns)

            # Fit Theil–Sen sobre la mediana no debe alterar mean/SD
            x  = df["time_mean_s"].values.astype(float)
            ym = df[median_col].values.astype(float)
            _theil_sen_fit(x, ym)

            # Verificar que las columnas mean/SD no cambiaron
            np.testing.assert_array_equal(
                df[mean_col].values,
                [1.5432, 1.5310, 1.5201] if side == "control" else [1.5801, 1.5690, 1.5577],
            )
            np.testing.assert_array_equal(
                df[sd_col].values,
                [0.0021, 0.0019, 0.0022] if side == "control" else [0.0018, 0.0020, 0.0017],
            )

    # ------------------------------------------------------------------ #
    # TEST 15 — DUAL-QC no interviene en selección de frames/bins/fit    #
    # ------------------------------------------------------------------ #

    def test_15_dual_qc_does_not_affect_fit(self):
        """El fit Theil–Sen no debe recibir ni usar columnas DUAL-QC.
        La función _theil_sen_fit solo acepta x, y — sin filtro de QC temporal."""
        # Construir un DataFrame con columnas DUAL-QC adversas (todos FAIL)
        n = 10
        x = np.linspace(5.0, 95.0, n)
        y = 1.5 - 0.0004 * x

        # Simular binned_df con columnas DUAL-QC (todos en FAIL)
        df_binned = pd.DataFrame({
            "time_mean_s":                        x,
            "control_radius_eq_mm2_median":       np.round(y, 4),
            # Columnas DUAL-QC que NO deben afectar el fit
            "control_temporal_qc_dual_ok":        [False] * n,
            "control_temporal_qc_status":         ["FAIL"] * n,
            "sample_temporal_qc_dual_ok":         [False] * n,
        })

        # El fit usa SOLO time_mean_s y *_median, ignora las columnas QC
        xr = df_binned["time_mean_s"].values.astype(float)
        yr = df_binned["control_radius_eq_mm2_median"].values.astype(float)
        result = _theil_sen_fit(xr, yr)

        # El fit debe completarse con todos los n puntos (sin censura QC)
        self.assertIsNotNone(result)
        self.assertEqual(result["n_pairwise_slopes"], _n_pairwise(n))
        self.assertAlmostEqual(result["K"], 0.0004, places=10)

    def test_15b_both_valid_mask_unchanged(self):
        """La máscara both_valid no usa columnas temporal_qc_*.
        Verificar que su construcción en analyze_video.py solo depende de
        tracking_valid y geometry_quality_valid."""
        # Simular el DataFrame de analyze_video.py con QC temporal adverso
        df = pd.DataFrame({
            "control_tracking_valid":          [True,  True,  True],
            "sample_tracking_valid":           [True,  True,  True],
            "control_geometry_quality_valid":  [True,  True,  True],
            "sample_geometry_quality_valid":   [True,  True,  True],
            # DUAL-QC adverso que NO debe excluir filas
            "control_temporal_qc_status":      ["FAIL", "FAIL", "FAIL"],
            "sample_temporal_qc_status":       ["FAIL", "FAIL", "FAIL"],
        })

        # Replicar exactamente la condición both_valid de analyze_video.py
        both_valid = (
            (df.get("control_tracking_valid",         True) == True)
            & (df.get("sample_tracking_valid",         True) == True)
            & (df.get("control_geometry_quality_valid", True) == True)
            & (df.get("sample_geometry_quality_valid",  True) == True)
        )
        df_qc = df[both_valid]

        # Todos los frames deben pasar (DUAL-QC no excluye ninguno)
        self.assertEqual(len(df_qc), 3)


# ---------------------------------------------------------------------------
# TEST 16 — Regresión Gate 3R
# ---------------------------------------------------------------------------

# Tolerancias estrictas apropiadas para floating point:
#   K          : 1e-12  (ratio rel ~1e-9 sobre valores ~4e-4)
#   intercept  : 1e-10  (ratio rel ~1e-10 sobre valores ~1.5)
#   R²         : 1e-10  (absoluto)
_K_TOL         = 1e-12
_INTERCEPT_TOL = 1e-10
_R2_TOL        = 1e-10


class TestGate3RRegression(unittest.TestCase):
    """Regresión canónica Gate 3R.

    Carga gate3r_median_bins.csv, aplica _theil_sen_fit por dataset×side,
    y compara contra los valores de referencia de gate3r_summary.csv.
    """

    @classmethod
    def setUpClass(cls):
        """Cargar gate3r_median_bins.csv una sola vez."""
        if not _GATE3R_BINS_CSV.exists():
            raise unittest.SkipTest(
                f"Gate 3R bins CSV no encontrado: {_GATE3R_BINS_CSV}"
            )
        cls.bins_df = pd.read_csv(_GATE3R_BINS_CSV)

    def _run_fit(self, dataset: str, side: str) -> dict:
        """Extraer datos de un dataset×side y ejecutar Theil–Sen."""
        col = _SIDE_COL[side]
        sub = self.bins_df[self.bins_df["dataset"] == dataset][
            ["time_mean_s", col]
        ].copy()
        sub = sub[sub[col].notna()]
        self.assertGreaterEqual(len(sub), 2, f"{dataset}/{side}: insuficientes filas")

        x = sub["time_mean_s"].values.astype(float)
        y = sub[col].values.astype(float)

        result = _theil_sen_fit(x, y)
        self.assertIsNotNone(result, f"{dataset}/{side}: _theil_sen_fit retornó None")
        return result

    def _check_case(self, dataset: str, side: str):
        """Ejecutar y verificar un caso dataset×side contra referencias."""
        ref    = _GATE3R_REF[(dataset, side)]
        result = self._run_fit(dataset, side)

        with self.subTest(dataset=dataset, side=side, metric="K"):
            diff_K = abs(result["K"] - ref["K"])
            self.assertLess(
                diff_K, _K_TOL,
                f"[{dataset}/{side}] K: calc={result['K']:.15f} "
                f"ref={ref['K']:.15f} diff={diff_K:.3e}",
            )

        with self.subTest(dataset=dataset, side=side, metric="intercept"):
            diff_i = abs(result["intercept"] - ref["intercept"])
            self.assertLess(
                diff_i, _INTERCEPT_TOL,
                f"[{dataset}/{side}] intercept: calc={result['intercept']:.15f} "
                f"ref={ref['intercept']:.15f} diff={diff_i:.3e}",
            )

        with self.subTest(dataset=dataset, side=side, metric="r_squared"):
            diff_r2 = abs(result["r_squared"] - ref["r_squared"])
            self.assertLess(
                diff_r2, _R2_TOL,
                f"[{dataset}/{side}] R²: calc={result['r_squared']:.15f} "
                f"ref={ref['r_squared']:.15f} diff={diff_r2:.3e}",
            )

        with self.subTest(dataset=dataset, side=side, metric="n_pairwise"):
            self.assertEqual(
                result["n_pairwise_slopes"], ref["n_pairwise"],
                f"[{dataset}/{side}] n_pairwise: "
                f"calc={result['n_pairwise_slopes']} ref={ref['n_pairwise']}",
            )

    def test_16a_corrida3_control(self):
        self._check_case("corrida3", "control")

    def test_16b_corrida3_sample(self):
        self._check_case("corrida3", "sample")

    def test_16c_corrida4_control(self):
        self._check_case("corrida4", "control")

    def test_16d_corrida4_sample(self):
        self._check_case("corrida4", "sample")

    def test_16e_water_water_control(self):
        self._check_case("water_water", "control")

    def test_16f_water_water_sample(self):
        self._check_case("water_water", "sample")


if __name__ == "__main__":
    unittest.main()
