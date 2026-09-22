"""Regression test for calibrated BODY-selector wiring."""

import unittest
from unittest.mock import patch

import numpy as np

from bubble_cv.detection import detect_bubbles


class TestPxToMmWiring(unittest.TestCase):

    def test_detect_bubbles_forwards_px_to_mm_to_both_rois(self):
        """Calibration must reach both ROI detectors."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # autospec is intentional:
        # it also verifies that _detect_drop_in_roi accepts px_to_mm.
        with patch(
            "bubble_cv.detection._detect_drop_in_roi",
            autospec=True,
            return_value=(None, {}),
        ) as roi_detector:
            detect_bubbles(frame, px_to_mm=34.08)

        self.assertEqual(roi_detector.call_count, 2)

        for call in roi_detector.call_args_list:
            self.assertIn("px_to_mm", call.kwargs)
            self.assertEqual(call.kwargs["px_to_mm"], 34.08)





class TestIndependentRoiSurvival(unittest.TestCase):
    """Un fallo unilateral de ROI no debe borrar la detección contralateral."""

    @staticmethod
    def _diag():
        return {
            "bodyellipse_fit_quality": {},
        }

    @staticmethod
    def _result(center_x):
        return (
            {
                "center_x": float(center_x),
            },
            object(),
        )

    def test_control_survives_when_sample_roi_fails(self):
        import numpy as np
        from types import SimpleNamespace
        from unittest.mock import patch
        import bubble_cv.detection as detection

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        control_result = self._result(150.0)
        sample_result = None

        def fake_build(props, cnt, px_to_mm, label):
            return SimpleNamespace(label=label)

        with patch.object(
            detection,
            "_detect_drop_in_roi",
            side_effect=[
                (control_result, self._diag()),
                (sample_result, self._diag()),
            ],
        ), patch.object(
            detection,
            "_build_detection",
            side_effect=fake_build,
        ):
            result = detection.detect_bubbles(
                frame,
                px_to_mm=34.08,
            )

        self.assertIsNotNone(result["control"])
        self.assertIsNone(result["sample"])
        self.assertEqual(result["control"].label, "control")

        self.assertIn("_audit", result)
        self.assertIn("control", result["_audit"])
        self.assertIn("sample", result["_audit"])

    def test_sample_survives_when_control_roi_fails(self):
        import numpy as np
        from types import SimpleNamespace
        from unittest.mock import patch
        import bubble_cv.detection as detection

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        control_result = None
        sample_result = self._result(450.0)

        def fake_build(props, cnt, px_to_mm, label):
            return SimpleNamespace(label=label)

        with patch.object(
            detection,
            "_detect_drop_in_roi",
            side_effect=[
                (control_result, self._diag()),
                (sample_result, self._diag()),
            ],
        ), patch.object(
            detection,
            "_build_detection",
            side_effect=fake_build,
        ):
            result = detection.detect_bubbles(
                frame,
                px_to_mm=34.08,
            )

        self.assertIsNone(result["control"])
        self.assertIsNotNone(result["sample"])
        self.assertEqual(result["sample"].label, "sample")

        self.assertIn("_audit", result)
        self.assertIn("control", result["_audit"])
        self.assertIn("sample", result["_audit"])


if __name__ == "__main__":
    unittest.main()
