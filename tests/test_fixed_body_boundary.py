"""
Tests del contrato de frontera física BODY fija.

La frontera opcional se expresa SIEMPRE en coordenadas Y globales
del frame. Cuando es None, el detector debe conservar exactamente
el comportamiento automático de RC3.
"""

import unittest
from unittest.mock import patch

import numpy as np

import bubble_cv.detection as detection


class TestFixedBodyBoundaryWiring(unittest.TestCase):
    """CONTROL y SAMPLE reciben referencias independientes."""

    def test_detect_bubbles_forwards_independent_fixed_boundaries(self):

        frame = np.zeros(
            (480, 640, 3),
            dtype=np.uint8,
        )

        with patch.object(
            detection,
            "_detect_drop_in_roi",
            autospec=True,
            return_value=(None, {}),
        ) as roi_detector:

            detection.detect_bubbles(
                frame,
                px_to_mm=34.08,
                control_body_start_y_global=61,
                sample_body_start_y_global=54,
            )

        self.assertEqual(
            roi_detector.call_count,
            2,
        )

        control_call = (
            roi_detector.call_args_list[0]
        )

        sample_call = (
            roi_detector.call_args_list[1]
        )

        self.assertEqual(
            control_call.kwargs[
                "fixed_body_start_y_global"
            ],
            61,
        )

        self.assertEqual(
            sample_call.kwargs[
                "fixed_body_start_y_global"
            ],
            54,
        )


    def test_default_none_is_forwarded_to_both_rois(self):
        """
        Sin referencia externa, ambos lados deben conservar
        explícitamente el modo automático.
        """

        frame = np.zeros(
            (480, 640, 3),
            dtype=np.uint8,
        )

        with patch.object(
            detection,
            "_detect_drop_in_roi",
            autospec=True,
            return_value=(None, {}),
        ) as roi_detector:

            detection.detect_bubbles(
                frame,
                px_to_mm=34.08,
            )

        self.assertEqual(
            roi_detector.call_count,
            2,
        )

        for call in (
            roi_detector.call_args_list
        ):
            self.assertIsNone(
                call.kwargs[
                    "fixed_body_start_y_global"
                ]
            )


class TestFixedBodyBoundaryCore(unittest.TestCase):
    """
    La referencia física debe sustituir BODY-start antes de
    seleccionar los puntos enviados a fitEllipse.
    """

    @staticmethod
    def _synthetic_setup():

        frame = np.zeros(
            (200, 200, 3),
            dtype=np.uint8,
        )

        # Componente rectangular de anchura constante.
        #
        # El algoritmo automático NO puede encontrar transición
        # neck -> body 1.6x, por lo que RC3 usaría fallback
        # Hough en y=40.
        #
        # Con frontera física y=30, el fit debe recibir
        # exactamente los puntos físicos con y >= 30.
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
        ).reshape(
            -1,
            1,
            2,
        )

        props = {
            "center_x": 42.0,
            "center_y": 40.0,
            "major_axis": 50.0,
            "minor_axis": 40.0,
            "angle_deg": 0.0,
            "semi_major": 25.0,
            "semi_minor": 20.0,
            "eccentricity": 0.6,
            "equiv_diameter_px":
                44.72135955,
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

        gray_roi = np.zeros(
            (100, 200),
            dtype=np.uint8,
        )

        mask_dyn = np.zeros(
            (82, 84),
            dtype=np.uint8,
        )

        return (
            frame,
            contour,
            props,
            fit_quality,
            gray_roi,
            mask_dyn,
        )


    def test_fixed_boundary_is_exact_cut_and_not_hough_fallback(self):

        (
            frame,
            contour,
            props,
            fit_quality,
            gray_roi,
            mask_dyn,
        ) = self._synthetic_setup()

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
                    [
                        [
                            [
                                100.0,
                                40.0,
                                25.0,
                            ]
                        ]
                    ],
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
                return_value=(
                    [contour],
                    None,
                ),
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

            result, diag = (
                detection._detect_drop_in_roi(
                    frame,
                    x0=0,
                    y0=0,
                    x1=200,
                    y1=100,
                    roi_label="sample",
                    frame_h=200,
                    px_to_mm=34.08,
                    fixed_body_start_y_global=30,
                )
            )

        self.assertIsNotNone(
            result
        )

        self.assertTrue(
            fit_mock.called
        )

        fitted_contour = (
            fit_mock.call_args.args[0]
        )

        fitted_points = (
            fitted_contour.reshape(
                -1,
                2,
            )
        )

        # La frontera solicitada es EXACTAMENTE 30.
        self.assertEqual(
            int(
                fitted_points[:, 1].min()
            ),
            30,
        )

        self.assertEqual(
            diag[
                "body_start_y_global"
            ],
            30,
        )

        # Una frontera explícita no es un
        # Hough fallback.
        self.assertFalse(
            diag[
                "body_fallback_used"
            ]
        )


    def test_fixed_boundary_outside_roi_is_rejected(self):
        """
        No clamp silencioso:
        una referencia fuera del ROI debe producir ValueError.
        """

        frame = np.zeros(
            (200, 200, 3),
            dtype=np.uint8,
        )

        with self.assertRaises(
            ValueError
        ):
            detection._detect_drop_in_roi(
                frame,
                x0=0,
                y0=0,
                x1=200,
                y1=100,
                roi_label="sample",
                frame_h=200,
                px_to_mm=34.08,
                fixed_body_start_y_global=100,
            )


    def test_negative_fixed_boundary_is_rejected(self):

        frame = np.zeros(
            (200, 200, 3),
            dtype=np.uint8,
        )

        with self.assertRaises(
            ValueError
        ):
            detection._detect_drop_in_roi(
                frame,
                x0=0,
                y0=0,
                x1=200,
                y1=100,
                roi_label="sample",
                frame_h=200,
                px_to_mm=34.08,
                fixed_body_start_y_global=-1,
            )


if __name__ == "__main__":
    unittest.main()
