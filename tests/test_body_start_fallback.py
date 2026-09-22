"""
Regression tests for the rc3 BODY fallback.

A failure to locate the neck->body transition must not, by itself,
erase a drop when Hough localisation and a usable physical contour exist.
"""

import unittest
from unittest.mock import patch

import numpy as np

import bubble_cv.detection as detection


class TestBodyStartFallback(unittest.TestCase):

    def test_body_start_failure_uses_hough_center_lower_arc(self):
        """
        Synthetic constant-width component:
        - Hough localises the drop.
        - A physical contour exists.
        - neck->body expansion cannot be found.
        - The detector must fall back to the physical contour arc at/below
          the Hough centre rather than returning None.
        """

        frame = np.zeros((200, 200, 3), dtype=np.uint8)

        # Constant-width component. Its first occupied rows are already
        # as wide as the rest, so the historical 1.6x expansion criterion
        # cannot produce body_start_y.
        contour = np.array(
            [
                [20, 10],
                [40, 10],
                [60, 10],
                [60, 20],
                [60, 30],
                [60, 40],
                [60, 50],
                [60, 60],
                [60, 70],
                [40, 70],
                [20, 70],
                [20, 60],
                [20, 50],
                [20, 40],
                [20, 30],
                [20, 20],
            ],
            dtype=np.int32,
        ).reshape(-1, 1, 2)

        # Expected ellipse from the fallback contour.
        # Dynamic crop starts at x=58, so local x=42 -> global x=100.
        props = {
            "center_x": 42.0,
            "center_y": 40.0,
            "major_axis": 50.0,
            "minor_axis": 40.0,
            "angle_deg": 0.0,
            "semi_major": 25.0,
            "semi_minor": 20.0,
            "eccentricity": 0.6,
            "equiv_diameter_px": 44.72135955,
        }

        fit_quality = {
            "fit_point_count": 10,
            "contour_area_px2": None,
            "ellipse_area_px2": 1570.8,
            "area_ratio": None,
            "iou": None,
            "residual_mean": 0.02,
            "residual_rmse": 0.03,
            "residual_p95": 0.05,
        }

        gray_roi = np.zeros((100, 200), dtype=np.uint8)
        mask_dyn = np.zeros((82, 84), dtype=np.uint8)

        with (
            patch.object(
                detection,
                "preprocess",
                return_value=gray_roi,
            ),
            patch.object(
                detection.cv2,
                "HoughCircles",
                return_value=np.array(
                    [[[100.0, 40.0, 25.0]]],
                    dtype=np.float32,
                ),
            ),
            patch.object(
                detection.cv2,
                "adaptiveThreshold",
                return_value=mask_dyn,
            ),
            patch.object(
                detection.cv2,
                "findContours",
                return_value=([contour], None),
            ),
            patch.object(
                detection,
                "_fit_single_ellipse",
                return_value=props,
            ) as fit_mock,
            patch.object(
                detection,
                "_compute_ellipse_fit_quality",
                return_value=fit_quality,
            ),
        ):
            result, diag = detection._detect_drop_in_roi(
                frame,
                x0=0,
                y0=0,
                x1=200,
                y1=100,
                roi_label="control",
                frame_h=200,
                px_to_mm=34.08,
            )

        # rc2 fails here: body_start_not_found -> result is None.
        self.assertIsNotNone(
            result,
            "body_start_not_found must not erase an otherwise usable drop",
        )

        # The fallback must use the physical lower arc, beginning at
        # the Hough centre y=40 in dynamic-crop coordinates.
        self.assertTrue(fit_mock.called)

        fitted_contour = fit_mock.call_args.args[0]
        fitted_points = fitted_contour.reshape(-1, 2)

        self.assertGreaterEqual(len(fitted_points), 5)
        self.assertGreaterEqual(
            fitted_points[:, 1].min(),
            40,
            "fallback must exclude contour points above the Hough centre",
        )

        self.assertTrue(diag["bodyellipse_used"])
        self.assertNotEqual(
            diag["bodyellipse_failure_reason"],
            "body_start_not_found",
        )


if __name__ == "__main__":
    unittest.main()
