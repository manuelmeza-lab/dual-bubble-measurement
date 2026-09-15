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


if __name__ == "__main__":
    unittest.main()
