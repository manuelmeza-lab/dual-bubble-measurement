import unittest

import pandas as pd

from bubble_cv.detection import BubbleDetection
from analyze_video import _side_valid_mask


class TestContourConsensusExport(unittest.TestCase):

    def test_detection_exports_consensus_fields(self):
        det = BubbleDetection()

        det.contour_consensus_applicable = True
        det.contour_consensus_valid = False
        det.contour_consensus_support_fraction = 0.4242
        det.contour_consensus_n_inliers = 28
        det.contour_consensus_n_points = 66
        det.contour_consensus_rejection_reason = "low_consensus"

        d = det.to_dict()

        self.assertEqual(
            d["contour_consensus_applicable"],
            True,
        )

        self.assertEqual(
            d["contour_consensus_valid"],
            False,
        )

        self.assertEqual(
            d["contour_consensus_support_fraction"],
            0.4242,
        )

        self.assertEqual(
            d["contour_consensus_n_inliers"],
            28,
        )

        self.assertEqual(
            d["contour_consensus_n_points"],
            66,
        )

        self.assertEqual(
            d["contour_consensus_rejection_reason"],
            "low_consensus",
        )


class TestContourConsensusAnalyticalValidity(unittest.TestCase):

    def test_low_consensus_vetoes_measurement(self):
        df = pd.DataFrame(
            {
                "sample_tracking_valid": [
                    True,
                    True,
                ],
                "sample_radius_eq_mm2": [
                    1.10,
                    2.35,
                ],
                "sample_contour_consensus_applicable": [
                    True,
                    True,
                ],
                "sample_contour_consensus_valid": [
                    True,
                    False,
                ],
            }
        )

        mask = _side_valid_mask(
            df,
            "sample",
        )

        self.assertEqual(
            mask.tolist(),
            [True, False],
        )

    def test_measurement_is_not_erased_by_qc(self):
        df = pd.DataFrame(
            {
                "sample_tracking_valid": [
                    True,
                ],
                "sample_radius_eq_mm2": [
                    2.3470,
                ],
                "sample_contour_consensus_applicable": [
                    True,
                ],
                "sample_contour_consensus_valid": [
                    False,
                ],
            }
        )

        mask = _side_valid_mask(
            df,
            "sample",
        )

        self.assertFalse(
            bool(mask.iloc[0])
        )

        # QC veta el análisis, pero conserva
        # la medición original para auditoría.
        self.assertEqual(
            df.loc[
                0,
                "sample_radius_eq_mm2",
            ],
            2.3470,
        )

    def test_non_applicable_consensus_does_not_veto(self):
        """Automatic BODY keeps historical RC4 analytical validity."""
        df = pd.DataFrame(
            {
                "control_tracking_valid": [
                    True,
                ],
                "control_radius_eq_mm2": [
                    1.10,
                ],
                "control_contour_consensus_applicable": [
                    False,
                ],
                "control_contour_consensus_valid": [
                    False,
                ],
            }
        )

        mask = _side_valid_mask(
            df,
            "control",
        )

        self.assertEqual(
            mask.tolist(),
            [True],
        )

    def test_legacy_dataframe_without_column_is_preserved(self):
        df = pd.DataFrame(
            {
                "sample_tracking_valid": [
                    True,
                    False,
                ],
                "sample_radius_eq_mm2": [
                    1.10,
                    1.20,
                ],
            }
        )

        mask = _side_valid_mask(
            df,
            "sample",
        )

        self.assertEqual(
            mask.tolist(),
            [True, False],
        )


if __name__ == "__main__":
    unittest.main()
