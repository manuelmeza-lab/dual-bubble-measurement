"""
FIX 7 — contrato del RMSE de bodyellipse como diagnóstico.

El residual RMSE y geometry_quality_valid deben seguir calculándose y
exportándose, pero un RMSE > 0.08 no debe, por sí solo, eliminar una
medición del análisis.

La validez analítica por lado depende de:
- tracking_valid == True
- radius_eq_mm2 disponible

geometry_quality_valid queda como metadata diagnóstica.
"""

import unittest
from types import SimpleNamespace

import pandas as pd

from analyze_video import (
    _bin_side_independently,
    _side_frame_counts,
    _side_valid_mask,
    apply_geometry_quality_gate,
)


class TestRmseDiagnosticContract(unittest.TestCase):

    def setUp(self):
        # t=0:
        #   Todo correcto.
        #
        # t=1:
        #   tracking físico correcto + medición presente,
        #   pero RMSE geométrico supera el threshold.
        #   FIX 7 exige conservarlo analíticamente.
        #
        # t=2:
        #   tracking_valid=False.
        #   Debe seguir excluido aunque exista medición.
        self.df = pd.DataFrame({
            "timestamp_s": [0.0, 1.0, 2.0],

            "control_detected": [
                True,
                True,
                True,
            ],

            "control_tracking_valid": [
                True,
                True,
                False,
            ],

            "control_geometry_quality_valid": [
                True,
                False,
                False,
            ],

            "control_geometry_quality_rejection_reason": [
                "",
                "bodyellipse_residual_rmse",
                "bodyellipse_residual_rmse",
            ],

            "control_bodyellipse_residual_rmse": [
                0.03,
                0.20,
                0.25,
            ],

            "control_radius_eq_mm2": [
                1.50,
                1.49,
                1.48,
            ],

            # Columnas mínimas adicionales requeridas por el binning.
            "control_equiv_diameter_mm": [
                2.40,
                2.39,
                2.38,
            ],
            "control_volume_mm3": [
                7.00,
                6.90,
                6.80,
            ],
            "control_radius_eq_mm": [
                1.2247,
                1.2207,
                1.2166,
            ],
            "control_eccentricity": [
                0.30,
                0.31,
                0.32,
            ],
        })

    def test_01_high_rmse_does_not_invalidate_analytical_measurement(self):
        mask = _side_valid_mask(self.df, "control")

        self.assertEqual(
            mask.tolist(),
            [True, True, False],
            "RMSE diagnostic failure must not veto an otherwise valid measurement",
        )

    def test_02_independent_binning_keeps_high_rmse_measurement(self):
        out = _bin_side_independently(
            self.df,
            side="control",
            bin_size_s=10.0,
        )

        self.assertEqual(len(out), 1)
        self.assertEqual(
            int(out.iloc[0]["n_points"]),
            2,
            "The high-RMSE t=1 point must remain in analytical binning",
        )

        self.assertEqual(
            out.iloc[0]["control_radius_eq_mm2_median"],
            1.495,
        )

    def test_03_frame_counts_do_not_call_rmse_diagnostic_a_qc_rejection(self):
        counts = _side_frame_counts(self.df, "control")

        self.assertEqual(counts["total_frames"], 3)
        self.assertEqual(counts["detected_frames"], 3)

        # t=0 and t=1 are analytically usable.
        self.assertEqual(counts["valid_frames"], 2)

        # Only t=2 is analytically rejected because tracking_valid=False.
        self.assertEqual(counts["qc_rejected_frames"], 1)
        self.assertEqual(counts["unusable_frames"], 1)

    def test_04_rmse_flag_is_still_computed_and_preserved(self):
        det = SimpleNamespace(
            bodyellipse_residual_rmse=0.20,
            geometry_quality_valid=None,
            geometry_quality_rejection_reason=None,
        )

        apply_geometry_quality_gate(det)

        # FIX 7 does NOT erase or falsify the diagnostic.
        self.assertFalse(det.geometry_quality_valid)
        self.assertEqual(
            det.geometry_quality_rejection_reason,
            "bodyellipse_residual_rmse",
        )
        self.assertEqual(det.bodyellipse_residual_rmse, 0.20)


class TestRmseDiagnosticEvaporationRate(unittest.TestCase):

    def test_05_high_rmse_measurement_participates_in_evaporation_rate(self):
        """
        t=1 has RMSE diagnostic failure but remains physically valid.
        Its volume must participate in dV/dt rather than being censored.
        """
        from analyze_video import _compute_side_evaporation_rate

        df = pd.DataFrame({
            "timestamp_s": [0.0, 1.0, 2.0],

            "control_tracking_valid": [
                True,
                True,
                False,
            ],

            "control_geometry_quality_valid": [
                True,
                False,
                False,
            ],

            "control_radius_eq_mm2": [
                1.50,
                1.49,
                1.48,
            ],

            "control_volume_mm3": [
                7.0,
                6.9,
                6.8,
            ],
        })

        rate = _compute_side_evaporation_rate(
            df,
            side="control",
        )

        # First valid point has no previous point -> NaN.
        self.assertTrue(pd.isna(rate.iloc[0]))

        # t=1 must survive despite geometry_quality_valid=False.
        # (6.9 - 7.0) / (1.0 - 0.0) = -0.1 mm^3/s
        self.assertAlmostEqual(
            float(rate.iloc[1]),
            -0.1,
            places=12,
        )

        # t=2 fails physical tracking and must remain excluded.
        self.assertTrue(pd.isna(rate.iloc[2]))


class TestRmseDiagnosticPairedValidity(unittest.TestCase):

    def test_06_paired_validity_is_intersection_without_rmse_veto(self):
        """
        La validez pareada debe ser la intersección de las validades
        analíticas individuales. geometry_quality_valid sigue siendo
        diagnóstico y no debe vetar por sí sola el par.
        """
        from analyze_video import _paired_valid_mask

        df = pd.DataFrame({
            "timestamp_s": [0.0, 1.0, 2.0, 3.0],

            "control_tracking_valid": [
                True,
                True,
                True,
                False,
            ],
            "control_geometry_quality_valid": [
                True,
                False,
                True,
                True,
            ],
            "control_radius_eq_mm2": [
                1.50,
                1.49,
                1.48,
                1.47,
            ],

            "sample_tracking_valid": [
                True,
                True,
                True,
                True,
            ],
            "sample_geometry_quality_valid": [
                True,
                True,
                False,
                True,
            ],
            "sample_radius_eq_mm2": [
                1.30,
                1.29,
                None,
                1.27,
            ],
        })

        paired = _paired_valid_mask(df)

        # t=0: ambos válidos.
        #
        # t=1: CONTROL tiene RMSE diagnostic failure, pero ambos lados
        #      son analíticamente válidos -> conservar.
        #
        # t=2: SAMPLE carece de radius_eq_mm2 -> excluir.
        #
        # t=3: CONTROL tracking_valid=False -> excluir.
        self.assertEqual(
            paired.tolist(),
            [True, True, False, False],
        )

if __name__ == "__main__":
    unittest.main()
