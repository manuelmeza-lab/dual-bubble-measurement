"""
tests/test_independent_dual_analysis.py

FIX 5 — contrato para análisis dual desacoplado.

Objetivo:
- la validez individual de una gota depende sólo de su propio QC;
- la validez pareada es la intersección de ambas;
- una gota ausente no invalida automáticamente a la otra;
- el binning individual puede utilizar información que no pertenece
  a la intersección pareada.
"""

import unittest

import pandas as pd

from analyze_video import _side_valid_mask


class TestIndependentDualAnalysisContract(unittest.TestCase):

    def setUp(self):
        # Seis timestamps.
        #
        # CONTROL:
        #   válido en 0,1,2,3,4
        #   ausente en 5
        #
        # SAMPLE:
        #   analíticamente válido en 0,1,3,4,5
        #   ausente en 2
        #   RMSE diagnóstico rechazado en 3, sin veto analítico
        #
        # Intersección válida:
        #   0,1,3,4
        self.df = pd.DataFrame({
            "timestamp_s": [0., 1., 2., 3., 4., 5.],

            "control_tracking_valid":
                [True, True, True, True, True, None],
            "control_geometry_quality_valid":
                [True, True, True, True, True, None],
            "control_radius_eq_mm2":
                [1.50, 1.49, 1.48, 1.47, 1.46, None],

            "sample_tracking_valid":
                [True, True, None, True, True, True],
            "sample_geometry_quality_valid":
                [True, True, None, False, True, True],
            "sample_radius_eq_mm2":
                [1.30, 1.29, None, 1.60, 1.27, 1.26],
        })

    def test_01_control_validity_is_independent(self):
        mask = _side_valid_mask(self.df, "control")

        self.assertEqual(
            mask.tolist(),
            [True, True, True, True, True, False],
        )

    def test_02_sample_validity_is_independent(self):
        mask = _side_valid_mask(self.df, "sample")

        self.assertEqual(
            mask.tolist(),
            [True, True, False, True, True, True],
        )

    def test_03_paired_validity_is_intersection(self):
        ctrl = _side_valid_mask(self.df, "control")
        samp = _side_valid_mask(self.df, "sample")

        paired = ctrl & samp

        self.assertEqual(
            paired.tolist(),
            [True, True, False, True, True, False],
        )

    def test_04_sample_failure_does_not_remove_control(self):
        ctrl = _side_valid_mask(self.df, "control")
        samp = _side_valid_mask(self.df, "sample")

        # t=2: sample ausente, pero control sigue siendo válido.
        self.assertTrue(bool(ctrl.iloc[2]))
        self.assertFalse(bool(samp.iloc[2]))

    def test_05_geometry_diagnostic_does_not_veto_measurement(self):
        ctrl = _side_valid_mask(self.df, "control")
        samp = _side_valid_mask(self.df, "sample")

        # t=3: SAMPLE tiene medición y tracking válido.
        # geometry_quality_valid=False se conserva como diagnóstico,
        # pero no veta la medición de SAMPLE ni afecta CONTROL.
        self.assertTrue(bool(ctrl.iloc[3]))
        self.assertTrue(bool(samp.iloc[3]))

    def test_06_independent_counts_differ_from_paired_count(self):
        ctrl = _side_valid_mask(self.df, "control")
        samp = _side_valid_mask(self.df, "sample")
        paired = ctrl & samp

        self.assertEqual(int(ctrl.sum()), 5)
        self.assertEqual(int(samp.sum()), 5)
        self.assertEqual(int(paired.sum()), 4)




class _FakeDetection:
    """Detección mínima para probar persistencia independiente."""

    def __init__(self, radius_eq_mm2):
        self._radius_eq_mm2 = radius_eq_mm2

    def to_dict(self):
        return {
            "label": "fake",
            "tracking_valid": True,
            "geometry_quality_valid": True,
            "radius_eq_mm2": self._radius_eq_mm2,
        }


class TestIndependentFramePersistence(unittest.TestCase):

    def test_07_control_survives_when_sample_missing(self):
        from analyze_video import _build_independent_frame_row

        ctrl = _FakeDetection(1.50)

        row = _build_independent_frame_row(
            frame_num=20,
            timestamp_s=2.0,
            ctrl_det=ctrl,
            samp_det=None,
        )

        self.assertEqual(row["frame_id"], 20)
        self.assertEqual(row["timestamp_s"], 2.0)

        self.assertTrue(row["control_detected"])
        self.assertFalse(row["sample_detected"])

        self.assertEqual(row["control_radius_eq_mm2"], 1.50)
        self.assertNotIn("sample_radius_eq_mm2", row)

    def test_08_sample_survives_when_control_missing(self):
        from analyze_video import _build_independent_frame_row

        samp = _FakeDetection(1.30)

        row = _build_independent_frame_row(
            frame_num=30,
            timestamp_s=3.0,
            ctrl_det=None,
            samp_det=samp,
        )

        self.assertEqual(row["frame_id"], 30)
        self.assertEqual(row["timestamp_s"], 3.0)

        self.assertFalse(row["control_detected"])
        self.assertTrue(row["sample_detected"])

        self.assertNotIn("control_radius_eq_mm2", row)
        self.assertEqual(row["sample_radius_eq_mm2"], 1.30)

    def test_09_timestamp_survives_when_both_missing(self):
        from analyze_video import _build_independent_frame_row

        row = _build_independent_frame_row(
            frame_num=40,
            timestamp_s=4.0,
            ctrl_det=None,
            samp_det=None,
        )

        self.assertEqual(row["frame_id"], 40)
        self.assertEqual(row["timestamp_s"], 4.0)

        self.assertFalse(row["control_detected"])
        self.assertFalse(row["sample_detected"])

        self.assertNotIn("control_radius_eq_mm2", row)
        self.assertNotIn("sample_radius_eq_mm2", row)


class TestIndependentFrameCounts(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "timestamp_s": [0., 1., 2., 3., 4., 5.],

            "control_detected":
                [True, True, True, True, True, False],
            "control_tracking_valid":
                [True, True, True, True, True, None],
            "control_geometry_quality_valid":
                [True, True, True, True, True, None],
            "control_radius_eq_mm2":
                [1.50, 1.49, 1.48, 1.47, 1.46, None],

            "sample_detected":
                [True, True, False, True, True, True],
            "sample_tracking_valid":
                [True, True, None, True, True, True],
            "sample_geometry_quality_valid":
                [True, True, None, False, True, True],
            "sample_radius_eq_mm2":
                [1.30, 1.29, None, 1.60, 1.27, 1.26],
        })

    def test_10_control_counts_are_independent(self):
        from analyze_video import _side_frame_counts

        counts = _side_frame_counts(self.df, "control")

        self.assertEqual(counts["total_frames"], 6)
        self.assertEqual(counts["detected_frames"], 5)
        self.assertEqual(counts["valid_frames"], 5)
        self.assertEqual(counts["missing_frames"], 1)
        self.assertEqual(counts["qc_rejected_frames"], 0)
        self.assertEqual(counts["unusable_frames"], 1)

    def test_11_sample_counts_keep_rmse_as_diagnostic(self):
        from analyze_video import _side_frame_counts

        counts = _side_frame_counts(self.df, "sample")

        self.assertEqual(counts["total_frames"], 6)
        self.assertEqual(counts["detected_frames"], 5)
        self.assertEqual(counts["valid_frames"], 5)
        self.assertEqual(counts["missing_frames"], 1)
        self.assertEqual(counts["qc_rejected_frames"], 0)
        self.assertEqual(counts["unusable_frames"], 1)



class TestIndependentBinning(unittest.TestCase):

    def setUp(self):
        # Dos bins de 10 s.
        #
        # CONTROL válido: t=0,1,2,10,11
        # SAMPLE válido:  t=0,1,10,11
        #
        # t=2 demuestra que un SAMPLE ausente no debe quitar CONTROL.
        # t=11 demuestra que geometry_quality_valid=False permanece como
        # diagnóstico y no elimina SAMPLE ni CONTROL del análisis.
        self.df = pd.DataFrame({
            "timestamp_s": [0., 1., 2., 10., 11.],

            "control_detected":
                [True, True, True, True, True],
            "control_tracking_valid":
                [True, True, True, True, True],
            "control_geometry_quality_valid":
                [True, True, True, True, True],
            "control_radius_eq_mm2":
                [1.50001, 1.49001, 1.48001, 1.40001, 1.39001],

            "sample_detected":
                [True, True, False, True, True],
            "sample_tracking_valid":
                [True, True, None, True, True],
            "sample_geometry_quality_valid":
                [True, True, None, True, False],
            "sample_radius_eq_mm2":
                [1.30001, 1.29001, None, 1.20001, 1.50001],
        })

    def test_12_control_binning_keeps_control_only_points(self):
        from analyze_video import _bin_side_independently

        out = _bin_side_independently(
            self.df,
            side="control",
            bin_size_s=10.0,
        )

        self.assertEqual(out["bin_id"].tolist(), [0, 1])
        self.assertEqual(out["n_points"].tolist(), [3, 2])

        self.assertEqual(
            out["control_radius_eq_mm2_median"].tolist(),
            [1.49, 1.395],
        )

    def test_13_sample_binning_uses_only_sample_validity(self):
        from analyze_video import _bin_side_independently

        out = _bin_side_independently(
            self.df,
            side="sample",
            bin_size_s=10.0,
        )

        self.assertEqual(out["bin_id"].tolist(), [0, 1])
        self.assertEqual(out["n_points"].tolist(), [2, 2])

        self.assertEqual(
            out["sample_radius_eq_mm2_median"].tolist(),
            [1.295, 1.35],
        )

    def test_14_median_is_rounded_to_four_decimals_before_fit(self):
        from analyze_video import _bin_side_independently

        out = _bin_side_independently(
            self.df,
            side="control",
            bin_size_s=10.0,
        )

        # Bin 0: median([1.50001, 1.49001, 1.48001]) = 1.49001
        # FIX 4 exige almacenar 1.4900 antes del Theil-Sen.
        self.assertEqual(
            out.loc[
                out["bin_id"] == 0,
                "control_radius_eq_mm2_median",
            ].iloc[0],
            1.49,
        )


if __name__ == "__main__":
    unittest.main()
